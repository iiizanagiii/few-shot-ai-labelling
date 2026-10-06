"""Few-shot box suggestion from a handful of labeled examples (v2).

Each labeled box is cropped, rotated (8 angles) and embedded with DINOv2; its
inner patch features form a support set. For a new image every patch is scored
by its best cosine match in the support set. That heatmap is averaged over an
object-sized window and local peaks become boxes of the example's size, so
touching or rotated objects no longer merge or fragment.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoModel

log = logging.getLogger("fewshot")

PATCH = 14
UPSAMPLE = 4
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CHUNK = 8192  # support rows per matmul, bounds memory


@dataclass
class Box:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0  # 1.0 = confirmed by the user

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def iou(self, o: "Box") -> float:
        iw = max(0.0, min(self.x2, o.x2) - max(self.x1, o.x1))
        ih = max(0.0, min(self.y2, o.y2) - max(self.y1, o.y1))
        inter = iw * ih
        return inter / (self.area() + o.area() - inter + 1e-9)


def read_rgb(path: Path) -> np.ndarray:
    """cv2.imread fails on non-ASCII Windows paths, so decode from bytes."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot decode image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class FewShotEngine:
    def __init__(self, model_name: str = "facebook/dinov2-small", long_side: int = 644):
        

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("load_model name=%s device=%s", model_name, self.device)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.long_side = long_side
        self._cache: dict[Path, tuple[np.ndarray, tuple[int, int]]] = {}
        self._support: dict[tuple, np.ndarray] = {}

    # ---- embedding ---------------------------------------------------------
    def _embed(self, batch: np.ndarray) -> np.ndarray:
        """uint8 (B, H, W, 3), H and W multiples of 14 -> normalised (B, gh, gw, C)."""
        b, h, w, _ = batch.shape
        gh, gw = h // PATCH, w // PATCH
        x = (batch.astype(np.float32) / 255 - MEAN) / STD
        with self._torch.inference_mode():
            t = self._torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).to(self.device)
            tokens = self.model(pixel_values=t).last_hidden_state[:, -gh * gw:]
        f = tokens.cpu().numpy().reshape(b, gh, gw, -1)
        return f / (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-9)

    def _features(self, path: Path) -> tuple[np.ndarray, tuple[int, int]]:
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

    # ---- support set -------------------------------------------------------
    def _box_support(self, path: Path, b: Box) -> np.ndarray:
        """Inner patch features of one box over rotated variants -> (N, C)."""
        key = (path, round(b.x1), round(b.y1), round(b.x2), round(b.y2))
        if key in self._support:
            return self._support[key]
        rgb = read_rgb(path)
        h, w = rgb.shape[:2]
        crop = rgb[int(b.y1): int(np.ceil(b.y2)), int(b.x1): int(np.ceil(b.x2))]
        s = self.long_side / max(h, w)
        nh = max(4 * PATCH, round(crop.shape[0] * s / PATCH) * PATCH)
        nw = max(4 * PATCH, round(crop.shape[1] * s / PATCH) * PATCH)
        crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        # Rotating a very elongated box would cut it off, so only flip it 180 degrees.
        roughly_square = 0.75 <= nw / nh <= 1.33
        angles = range(0, 360, 45) if roughly_square else (0, 180)
        variants = [
            cv2.warpAffine(crop, cv2.getRotationMatrix2D((nw / 2, nh / 2), a, 1.0), (nw, nh),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            for a in angles
        ]
        f = self._embed(np.stack(variants))
        if f.shape[1] >= 5 and f.shape[2] >= 5:
            f = f[:, 1:-1, 1:-1]  # drop the outer ring: mostly background / edge mixes
        sup = f.reshape(-1, f.shape[-1])
        self._support[key] = sup
        log.info("support image=%s variants=%d patches=%d", path.name, len(variants), len(sup))
        return sup

    def support(self, ann: dict[Path, list[Box]]) -> dict[int, tuple[np.ndarray, tuple[float, float]]]:
        """Per class: stacked support features and median example size (w, h) in pixels."""
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

    # ---- suggestion --------------------------------------------------------
    def suggest(
        self, path: Path, ann: dict[Path, list[Box]], thresh: float,
        core: float = 0.6, nms_iou: float = 0.3, max_cands: int = 200,
    ) -> list[Box]:
        sup = self.support(ann)
        if not sup:
            return []
        f, (w, h) = self._features(path)
        gh, gw = f.shape[:2]
        flat = f.reshape(-1, f.shape[-1])
        sx, sy = w / (gw * UPSAMPLE), h / (gh * UPSAMPLE)  # original px per heatmap px

        cands: list[Box] = []
        for c, (feat, (bw, bh)) in sup.items():
            sim = np.full(len(flat), -1.0, np.float32)
            for i in range(0, len(feat), CHUNK):
                sim = np.maximum(sim, (flat @ feat[i: i + CHUNK].T).max(1))
            up = cv2.resize(sim.reshape(gh, gw), (gw * UPSAMPLE, gh * UPSAMPLE), interpolation=cv2.INTER_LINEAR)
            # Mean score over the object's core (60% of its size) rejects thin edge matches.
            kw, kh = max(3, int(bw * core / sx) | 1), max(3, int(bh * core / sy) | 1)
            score = cv2.boxFilter(up, -1, (kw, kh), borderType=cv2.BORDER_REPLICATE)
            nk = (max(3, int(bw * 0.5 / sx) | 1), max(3, int(bh * 0.5 / sy) | 1))
            local_max = score >= cv2.dilate(score, np.ones((nk[1], nk[0]), np.uint8))
            ys, xs = np.nonzero(local_max & (score >= thresh))
            for y, x in zip(ys, xs):
                cx, cy = (x + 0.5) * sx, (y + 0.5) * sy
                cands.append(Box(c, max(0, cx - bw / 2), max(0, cy - bh / 2),
                                 min(w, cx + bw / 2), min(h, cy + bh / 2), float(score[y, x])))

        cands.sort(key=lambda b: -b.score)
        existing = ann.get(path, [])
        out: list[Box] = []
        for b in cands[:max_cands]:
            if any(b.iou(k) > nms_iou for k in out) or any(b.iou(e) > nms_iou for e in existing):
                continue
            out.append(b)
        log.info("suggest image=%s thresh=%.2f candidates=%d kept=%d", path.name, thresh, len(cands), len(out))
        return out