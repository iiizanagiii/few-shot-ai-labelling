"""Few-shot box suggestion, Matcher style: DINOv2 finds, SAM refines.

Overview
--------
Given a handful of boxes you have drawn (the "support" examples), this module
proposes boxes for the same kind of object in other images. There is no
training: all models are frozen and everything is done by feature matching.

The pipeline has two stages.

Stage 1, find candidates (DINOv2)
    1. Every image is resized and embedded with DINOv2, a Vision Transformer that
       outputs one L2-normalised feature vector per 14x14-pixel patch.
    2. Each labeled box is cropped, rotated in 45 degree steps (so stripe or
       texture orientation does not matter) and embedded the same way. The inner
       patch vectors of all rotations form that box's support set.
    3. For a new image, every patch is scored by its best cosine similarity to any
       support patch of a class (nearest-neighbour patch matching). This gives a
       heatmap per class.
    4. The heatmap is upsampled, averaged over an object-sized window (a mean
       filter), and local maxima above a threshold become candidate boxes of the
       median example size. Overlapping candidates are removed (NMS).

Stage 2, refine candidates (SAM)
    5. Each candidate box is used as a box prompt for the Segment Anything Model.
       SAM returns a mask, and the mask's tight bounding box replaces the rough
       window. Two passes are run: the first with the window box, the second with
       the tight box from the first, which is a better prompt.
    6. Each mask is verified. It is rejected if its box area is far from the
       example size, or if the mean DINOv2 similarity inside the mask is low. This
       drops SAM results that cover background or only a fragment.
    7. Surviving boxes are de-duplicated and returned sorted by score.

If SAM cannot be loaded (offline, old ``transformers``) the engine logs
``sam_unavailable`` and returns the stage 1 window boxes instead.

Coordinate conventions
----------------------
All ``Box`` coordinates are in original image pixels with the origin at the top
left. ``x2``/``y2`` are exclusive edges. Feature grids are (rows, cols) =
(gh, gw) patches of the resized image, never of the original.

Dependencies
------------
``numpy``, ``opencv-python-headless``, ``torch``, ``transformers`` (provides
DINOv2 via ``AutoModel`` and SAM via ``SamModel``/``SamProcessor``).
``torch`` and ``transformers`` are imported lazily inside ``FewShotEngine`` so
that importing this module (for ``Box``) stays fast.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoModel

log = logging.getLogger("fewshot")

PATCH = 14  # DINOv2 patch size in pixels; resized images are multiples of this
UPSAMPLE = (
    4  # heatmap upsampling per patch, smooths box positions between patch centres
)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet mean used by DINOv2
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)  # ImageNet std used by DINOv2
CHUNK = 8192  # support rows per matmul, bounds memory when many examples exist
SAM_BATCH = 8  # boxes decoded per SAM call, bounds full-resolution mask memory
SAM_CACHE = 6  # number of images whose SAM embeddings stay cached


@dataclass
class Box:
    """An axis-aligned bounding box with a class id and a confidence score.

    Attributes:
        cls: Zero-based class index.
        x1: Left edge in original image pixels.
        y1: Top edge in original image pixels.
        x2: Right edge in original image pixels (exclusive).
        y2: Bottom edge in original image pixels (exclusive).
        score: Confidence. ``1.0`` means confirmed by the user; suggestions carry
            the mean similarity (roughly 0.5 to 0.9) computed by the engine.
    """

    cls: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0  # 1.0 = confirmed by the user

    def area(self) -> float:
        """Return the box area in square pixels, or 0 for an empty/inverted box."""
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def iou(self, o: "Box") -> float:
        """Return intersection-over-union with another box, in the range [0, 1].

        Args:
            o: The other box. Classes are ignored; only geometry is compared.
        """
        iw = max(0.0, min(self.x2, o.x2) - max(self.x1, o.x1))
        ih = max(0.0, min(self.y2, o.y2) - max(self.y1, o.y1))
        inter = iw * ih
        return inter / (self.area() + o.area() - inter + 1e-9)


def read_rgb(path: Path) -> np.ndarray:
    """Read an image file as an RGB ``uint8`` array of shape (H, W, 3).

    ``cv2.imread`` cannot open paths containing non-ASCII characters on Windows,
    so the file is read as raw bytes with NumPy and decoded with ``cv2.imdecode``.

    Args:
        path: Image file path.

    Returns:
        RGB image array.

    Raises:
        ValueError: If the file cannot be decoded as an image.
    """
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot decode image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class FewShotEngine:
    """Few-shot object suggester combining DINOv2 matching and SAM refinement.

    The engine is stateful: it caches DINOv2 features per image, support features
    per example box, and SAM embeddings for recently viewed images. This makes
    repeated calls (for example while dragging a threshold slider or adding one
    more example) cheap, because only the matching and prompt decoding are redone.

    Typical usage::

        engine = FewShotEngine()
        ann = {Path("a.jpg"): [Box(cls=0, x1=10, y1=20, x2=110, y2=130)]}
        boxes = engine.suggest(Path("b.jpg"), ann, thresh=0.6)

    Attributes:
        device: ``"cuda"`` if available, otherwise ``"cpu"``.
        model: The DINOv2 model in eval mode.
        sam: The SAM model, or ``None`` if disabled or unavailable.
        sam_proc: The matching ``SamProcessor``, or ``None``.
        long_side: Target size in pixels of the longer image side before embedding.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        sam_name: str = "facebook/sam-vit-base",
        use_sam: bool = True,
        long_side: int = 644,
    ):
        """Load DINOv2 and, optionally, SAM.

        Weights are downloaded from the Hugging Face Hub on first use and cached
        locally afterwards. ``torch`` is imported here (not at module level) so
        importing this module does not pay the load cost.

        Args:
            model_name: Hugging Face id of the DINOv2 model. Larger variants
                (``dinov2-base``, ``dinov2-large``) give better features but are
                slower and use more memory.
            sam_name: Hugging Face id of the SAM model (``sam-vit-base``,
                ``sam-vit-large``, ``sam-vit-huge``). Larger is more accurate and
                slower.
            use_sam: If ``False``, skip loading SAM and return window boxes.
            long_side: Resize each image so its longer side is about this many
                pixels (rounded to a multiple of 14). More pixels means a finer
                patch grid and slower embedding.
        """
        

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("load_model name=%s device=%s", model_name, self.device)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.long_side = long_side
        # path -> (patch features (gh, gw, C), original size (w, h))
        self._cache: dict[Path, tuple[np.ndarray, tuple[int, int]]] = {}
        # (path, rounded box coords) -> support patch features (N, C)
        self._support: dict[tuple, np.ndarray] = {}

        self.sam = self.sam_proc = None
        # LRU cache: path -> (embedding, rgb, original_sizes, reshaped_input_sizes)
        self._sam_emb: OrderedDict[Path, tuple] = OrderedDict()
        if use_sam:
            try:
                from transformers import SamModel, SamProcessor

                log.info("load_model name=%s device=%s", sam_name, self.device)
                self.sam_proc = SamProcessor.from_pretrained(sam_name)
                self.sam = SamModel.from_pretrained(sam_name).to(self.device).eval()
            except Exception:  # offline, missing weights, old transformers...
                log.exception("sam_unavailable, falling back to window boxes")
                self.sam = self.sam_proc = None

    # ---- DINOv2 embedding ----------------------------------------------------
    def _embed(self, batch: np.ndarray) -> np.ndarray:
        """Embed a batch of images into L2-normalised DINOv2 patch features.

        Args:
            batch: ``uint8`` array of shape (B, H, W, 3), RGB. ``H`` and ``W`` must
                be multiples of 14 so the image divides evenly into patches.

        Returns:
            ``float32`` array of shape (B, gh, gw, C) with ``gh = H / 14``,
            ``gw = W / 14`` and ``C`` the model's hidden size (384 for ``small``).
            Each patch vector has unit length, so a dot product between two
            vectors is their cosine similarity.

        Note:
            The model's output also contains a global CLS token (and possibly
            register tokens). Only the last ``gh * gw`` tokens are kept, which are
            always the patch tokens.
        """
        b, h, w, _ = batch.shape
        gh, gw = h // PATCH, w // PATCH
        x = (batch.astype(np.float32) / 255 - MEAN) / STD
        with self._torch.inference_mode():
            t = self._torch.from_numpy(
                np.ascontiguousarray(x.transpose(0, 3, 1, 2))
            ).to(self.device)
            tokens = self.model(pixel_values=t).last_hidden_state[:, -gh * gw :]
        f = tokens.cpu().numpy().reshape(b, gh, gw, -1)
        return f / (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-9)

    def _features(self, path: Path) -> tuple[np.ndarray, tuple[int, int]]:
        """Return cached patch features for a whole image, computing them if needed.

        The image is resized so its longer side is ``long_side`` pixels, with both
        sides rounded to multiples of 14, then embedded once.

        Args:
            path: Image file path (also the cache key).

        Returns:
            A tuple ``(features, (w, h))`` where ``features`` has shape
            (gh, gw, C) and ``(w, h)`` is the original image size in pixels, needed
            later to map grid positions back to original coordinates.
        """
        if path in self._cache:
            return self._cache[path]
        rgb = read_rgb(path)
        h, w = rgb.shape[:2]
        s = self.long_side / max(h, w)
        nh = max(PATCH, round(h * s / PATCH) * PATCH)
        nw = max(PATCH, round(w * s / PATCH) * PATCH)
        img = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        self._cache[path] = (self._embed(img[None])[0], (w, h))
        log.info("embed image=%s grid=%dx%d", path.name, nw // PATCH, nh // PATCH)
        return self._cache[path]

    # ---- support set ---------------------------------------------------------
    def _box_support(self, path: Path, b: Box) -> np.ndarray:
        """Build the support feature set for one labeled box.

        Steps: crop the box, rescale it with the same factor used for full images
        (so the object appears at the same scale as in query features), make
        rotated variants, embed them in one batch, and keep the inner patches.

        * Variants: eight rotations (0, 45, ..., 315 degrees) for roughly square
          boxes (aspect ratio 0.75 to 1.33). Very elongated boxes would be cut off
          by rotation, so only 0 and 180 degrees are used. Borders are replicated
          to avoid black corners that would pollute the features.
        * Inner patches: when the grid is at least 5x5, the outer ring of patches
          is dropped because it mostly mixes object edge with background.

        Args:
            path: Image the box was drawn on.
            b: The labeled box, in original pixels.

        Returns:
            Array of shape (N, C) of unit-length patch vectors, cached by image and
            rounded box coordinates.
        """
        key = (path, round(b.x1), round(b.y1), round(b.x2), round(b.y2))
        if key in self._support:
            return self._support[key]
        rgb = read_rgb(path)
        h, w = rgb.shape[:2]
        crop = rgb[int(b.y1) : int(np.ceil(b.y2)), int(b.x1) : int(np.ceil(b.x2))]
        s = self.long_side / max(h, w)
        nh = max(4 * PATCH, round(crop.shape[0] * s / PATCH) * PATCH)
        nw = max(4 * PATCH, round(crop.shape[1] * s / PATCH) * PATCH)
        crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        # Rotating a very elongated box would cut it off, so only flip it 180 degrees.
        roughly_square = 0.75 <= nw / nh <= 1.33
        angles = range(0, 360, 45) if roughly_square else (0, 180)
        variants = [
            cv2.warpAffine(
                crop,
                cv2.getRotationMatrix2D((nw / 2, nh / 2), a, 1.0),
                (nw, nh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            for a in angles
        ]
        f = self._embed(np.stack(variants))
        if f.shape[1] >= 5 and f.shape[2] >= 5:
            f = f[:, 1:-1, 1:-1]  # drop the outer ring: mostly background / edge mixes
        sup = f.reshape(-1, f.shape[-1])
        self._support[key] = sup
        log.info(
            "support image=%s variants=%d patches=%d",
            path.name,
            len(variants),
            len(sup),
        )
        return sup

    def support(
        self, ann: dict[Path, list[Box]]
    ) -> dict[int, tuple[np.ndarray, tuple[float, float]]]:
        """Aggregate all labeled boxes into one support set per class.

        Args:
            ann: Mapping from image path to the confirmed boxes on that image.
                Images with no boxes are ignored.

        Returns:
            ``{class_id: (features, (w, h))}`` where ``features`` is the
            concatenation of every box's support patches, shape (N, C), and
            ``(w, h)`` is the median example size in original pixels. The median
            size later defines the window used to find candidate objects.
        """
        feats: dict[int, list[np.ndarray]] = {}
        sizes: dict[int, list[tuple[float, float]]] = {}
        for path, boxes in ann.items():
            for b in boxes:
                feats.setdefault(b.cls, []).append(self._box_support(path, b))
                sizes.setdefault(b.cls, []).append((b.x2 - b.x1, b.y2 - b.y1))
        return {
            c: (np.concatenate(feats[c]), tuple(np.median(np.array(sizes[c]), axis=0)))
            for c in feats
        }

    # ---- SAM -------------------------------------------------------------------
    def _sam_embedding(self, path: Path) -> tuple:
        """Return the SAM image embedding for ``path``, computing it if not cached.

        The heavy SAM image encoder runs once per image. Prompt decoding is cheap,
        so caching the embedding keeps slider changes and re-prompting fast. The
        cache holds the most recent ``SAM_CACHE`` images (least recently used are
        evicted).

        Args:
            path: Image file path.

        Returns:
            ``(embedding, rgb, original_sizes, reshaped_input_sizes)``. The sizes
            are the tensors ``SamProcessor`` needs to map masks back to the
            original resolution.
        """
        if path in self._sam_emb:
            self._sam_emb.move_to_end(path)
            return self._sam_emb[path]
        rgb = read_rgb(path)
        inputs = self.sam_proc(rgb, return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            emb = self.sam.get_image_embeddings(inputs["pixel_values"])
        self._sam_emb[path] = (
            emb,
            rgb,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
        )
        while len(self._sam_emb) > SAM_CACHE:
            self._sam_emb.popitem(last=False)
        log.info("sam_embed image=%s", path.name)
        return self._sam_emb[path]

    def _sam_masks(self, path: Path, boxes: list[Box]) -> np.ndarray:
        """Segment the object inside each box using SAM box prompts.

        Uses ``multimask_output=False`` so SAM returns the single best mask per
        box, which is the recommended mode for box prompts.

        Args:
            path: Image file path.
            boxes: Prompt boxes in original pixels. Keep the list short
                (``SAM_BATCH``) because every mask is returned at full resolution.

        Returns:
            Boolean array of shape (n, H, W) at the original image resolution,
            one mask per input box in the same order.
        """
        emb, rgb, _, _ = self._sam_embedding(path)
        inputs = self.sam_proc(
            rgb,
            input_boxes=[[[b.x1, b.y1, b.x2, b.y2] for b in boxes]],
            return_tensors="pt",
        )
        with self._torch.inference_mode():
            out = self.sam(
                image_embeddings=emb,
                input_boxes=inputs["input_boxes"].to(self.device),
                multimask_output=False,
            )
        masks = self.sam_proc.image_processor.post_process_masks(
            out.pred_masks.cpu(),
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
        )[0]  # (n, 1, H, W) bool
        return masks[:, 0].numpy()

    def _refine(
        self,
        path: Path,
        boxes: list[Box],
        ups: dict[int, np.ndarray],
        sizes: dict[int, tuple[float, float]],
        verify: float,
        passes: int = 2,
    ) -> list[Box]:
        """Replace rough window boxes with tight, verified SAM-mask boxes.

        Each pass prompts SAM with the current boxes (in chunks of ``SAM_BATCH``),
        converts each mask to its tight bounding box, and applies two checks:

        1. Size check: the tight box area must be between 0.35x and 2.5x the
           median example area of its class. This rejects fragments and
           whole-background masks.
        2. Similarity check: the mean DINOv2 similarity inside the mask must be at
           least ``verify``. This rejects masks over things that do not look like
           the examples.

        Survivors become the prompts of the next pass. Prompting with a tight box
        gives SAM a better cue than the original fixed-size window, so the second
        pass usually improves the fit.

        Args:
            path: Image file path.
            boxes: Candidate boxes from stage 1.
            ups: Per-class upsampled similarity heatmaps, shape (gh*4, gw*4).
            sizes: Per-class median example size ``(w, h)`` in original pixels.
            verify: Minimum mean in-mask similarity to keep a result.
            passes: Number of prompt passes.

        Returns:
            Refined boxes whose ``score`` is the mean in-mask similarity. Empty if
            nothing survives.
        """
        cur = boxes
        for _ in range(passes):
            nxt: list[Box] = []
            for i in range(0, len(cur), SAM_BATCH):
                chunk = cur[i : i + SAM_BATCH]
                for b, m in zip(chunk, self._sam_masks(path, chunk)):
                    cols, rows = np.flatnonzero(m.any(0)), np.flatnonzero(m.any(1))
                    if len(cols) == 0:
                        continue
                    tight = Box(
                        b.cls,
                        float(cols[0]),
                        float(rows[0]),
                        float(cols[-1] + 1),
                        float(rows[-1] + 1),
                    )
                    bw, bh = sizes[b.cls]
                    if not 0.35 <= tight.area() / (bw * bh + 1e-9) <= 2.5:
                        continue  # SAM grabbed a fragment or the background
                    up = ups[b.cls]
                    small = cv2.resize(
                        m.astype(np.uint8),
                        (up.shape[1], up.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    if not small.any():
                        continue
                    tight.score = float(
                        up[small > 0].mean()
                    )  # DINOv2 agreement inside the mask
                    if tight.score >= verify:
                        nxt.append(tight)
            cur = nxt
            if not cur:
                break
        return cur

    # ---- suggestion ------------------------------------------------------------
    def suggest(
        self,
        path: Path,
        ann: dict[Path, list[Box]],
        thresh: float,
        refine: bool = True,
        core: float = 0.6,
        nms_iou: float = 0.3,
        max_cands: int = 200,
        max_refine: int = 40,
    ) -> list[Box]:
        """Propose boxes for objects like the labeled examples in one image.

        Algorithm:

        1. Build the per-class support set from ``ann`` (see ``support``).
        2. For each class, score every patch of the query image by its maximum
           cosine similarity to any support patch, then upsample that map 4x.
        3. Average the map over a window of ``core`` times the median example size.
           A real object scores high over its whole core, while thin edge matches
           or partial objects do not.
        4. Keep local maxima of that score at or above ``thresh``. Each becomes a
           box of the median example size centred on the peak.
        5. Sort by score and greedily drop boxes that overlap a better one or an
           existing user box (IoU above ``nms_iou``).
        6. If ``refine`` is on and SAM is loaded, refine with SAM (see ``_refine``)
           and de-duplicate again at IoU 0.5; otherwise return the window boxes.

        Args:
            path: Image to find objects in.
            ann: Confirmed boxes per image, used as examples. Boxes already
                present on ``path`` are also used as examples and are never
                suggested again.
            thresh: Minimum window-mean cosine similarity for a candidate. About
                0.6 is a reasonable start; raise it to cut false positives, lower
                it to find more. SAM results are verified against ``thresh - 0.1``.
            refine: Use SAM to tighten boxes. Ignored if SAM is unavailable.
            core: Fraction of the example size used as the averaging window.
            nms_iou: IoU above which overlapping window candidates are dropped.
            max_cands: Cap on candidates considered before overlap removal.
            max_refine: Cap on boxes sent to SAM, to bound latency.

        Returns:
            Suggested boxes sorted by descending score. ``Box.score`` is the window
            similarity (no SAM) or the mean in-mask similarity (with SAM). Empty if
            there are no examples in ``ann``.
        """
        sup = self.support(ann)
        if not sup:
            return []
        f, (w, h) = self._features(path)
        gh, gw = f.shape[:2]
        flat = f.reshape(-1, f.shape[-1])
        sx, sy = w / (gw * UPSAMPLE), h / (gh * UPSAMPLE)  # original px per heatmap px

        cands: list[Box] = []
        ups: dict[int, np.ndarray] = {}
        sizes: dict[int, tuple[float, float]] = {}
        for c, (feat, (bw, bh)) in sup.items():
            sim = np.full(len(flat), -1.0, np.float32)
            for i in range(0, len(feat), CHUNK):
                sim = np.maximum(sim, (flat @ feat[i : i + CHUNK].T).max(1))
            up = cv2.resize(
                sim.reshape(gh, gw),
                (gw * UPSAMPLE, gh * UPSAMPLE),
                interpolation=cv2.INTER_LINEAR,
            )
            ups[c], sizes[c] = up, (bw, bh)
            # Mean score over the object's core (60% of its size) rejects thin edge matches.
            kw, kh = max(3, int(bw * core / sx) | 1), max(3, int(bh * core / sy) | 1)
            score = cv2.boxFilter(up, -1, (kw, kh), borderType=cv2.BORDER_REPLICATE)
            nk = (max(3, int(bw * 0.5 / sx) | 1), max(3, int(bh * 0.5 / sy) | 1))
            local_max = score >= cv2.dilate(score, np.ones((nk[1], nk[0]), np.uint8))
            ys, xs = np.nonzero(local_max & (score >= thresh))
            for y, x in zip(ys, xs):
                cx, cy = (x + 0.5) * sx, (y + 0.5) * sy
                cands.append(
                    Box(
                        c,
                        max(0, cx - bw / 2),
                        max(0, cy - bh / 2),
                        min(w, cx + bw / 2),
                        min(h, cy + bh / 2),
                        float(score[y, x]),
                    )
                )

        cands.sort(key=lambda b: -b.score)
        existing = ann.get(path, [])
        windows: list[Box] = []
        for b in cands[:max_cands]:
            if any(b.iou(k) > nms_iou for k in windows) or any(
                b.iou(e) > nms_iou for e in existing
            ):
                continue
            windows.append(b)

        if not (refine and self.sam is not None and windows):
            log.info(
                "suggest image=%s thresh=%.2f candidates=%d kept=%d sam=off",
                path.name,
                thresh,
                len(cands),
                len(windows),
            )
            return windows

        # Matcher-style second stage: SAM masks verified by DINOv2 similarity.
        refined = self._refine(
            path, windows[:max_refine], ups, sizes, verify=max(0.0, thresh - 0.1)
        )
        refined.sort(key=lambda b: -b.score)
        out: list[Box] = []
        for b in refined:
            if any(b.iou(k) > 0.5 for k in out) or any(
                b.iou(e) > 0.5 for e in existing
            ):
                continue
            out.append(b)
        log.info(
            "suggest image=%s thresh=%.2f candidates=%d windows=%d kept=%d sam=on",
            path.name,
            thresh,
            len(cands),
            len(windows),
            len(out),
        )
        return out
