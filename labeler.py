"""PyQt5 few-shot labeling app. Draw a few boxes, let the model suggest the rest.

Mouse:  drag = new box | click dashed box = accept it | right-click = delete box
Keys:   1-9 class | Left/Right image | S suggest | Enter accept all | Ctrl+S save
Output: <image folder>/labels/<name>.txt (YOLO format) + classes.txt
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from few_shot import Box

from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImageReader, QKeySequence, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QShortcut,
    QSlider,
    QVBoxLayout,
    QWidget,
)


log = logging.getLogger("labeler")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def class_color(c: int) -> QColor:
    return QColor.fromHsv((c * 47) % 360, 220, 255)


class Canvas(QWidget):
    changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.pix: QPixmap | None = None
        self.boxes: list[Box] = []  # confirmed (same list object as MainWindow.ann[path])
        self.preds: list[Box] = []  # suggestions
        self.names: list[str] = []
        self.cur_cls = 0
        self._p0 = self._p1 = None
        self.setMinimumSize(640, 480)
        self.setCursor(Qt.CrossCursor)

    def set_image(self, pix: QPixmap, boxes: list[Box]) -> None:
        self.pix, self.boxes, self.preds = pix, boxes, []
        self.update()

    def _name(self, c: int) -> str:
        return self.names[c] if c < len(self.names) else str(c)

    def _view(self) -> tuple[float, float, float]:
        s = min(self.width() / self.pix.width(), self.height() / self.pix.height())
        return s, (self.width() - self.pix.width() * s) / 2, (self.height() - self.pix.height() * s) / 2

    def _to_img(self, p) -> tuple[float, float]:
        s, ox, oy = self._view()
        return (min(max((p.x() - ox) / s, 0), self.pix.width()),
                min(max((p.y() - oy) / s, 0), self.pix.height()))

    def _rect(self, b: Box) -> QRectF:
        s, ox, oy = self._view()
        return QRectF(ox + b.x1 * s, oy + b.y1 * s, (b.x2 - b.x1) * s, (b.y2 - b.y1) * s)

    def _hit(self, items: list[Box], pos) -> int | None:
        for i in range(len(items) - 1, -1, -1):
            if self._rect(items[i]).contains(QPointF(pos)):
                return i
        return None

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(30, 30, 30))
        if self.pix is None:
            return
        s, ox, oy = self._view()
        p.drawPixmap(QRectF(ox, oy, self.pix.width() * s, self.pix.height() * s),
                     self.pix, QRectF(self.pix.rect()))
        for items, style in ((self.boxes, Qt.SolidLine), (self.preds, Qt.DashLine)):
            for b in items:
                p.setPen(QPen(class_color(b.cls), 2, style))
                r = self._rect(b)
                p.drawRect(r)
                tag = self._name(b.cls) if items is self.boxes else f"{self._name(b.cls)} {b.score:.2f}"
                p.drawText(r.topLeft() + QPointF(3, -4), tag)
        if self._p0 is not None and self._p1 is not None:
            p.setPen(QPen(Qt.white, 1, Qt.DashLine))
            p.drawRect(QRectF(QPointF(self._p0), QPointF(self._p1)).normalized())

    def mousePressEvent(self, e) -> None:
        if self.pix is None:
            return
        if e.button() == Qt.LeftButton:
            self._p0 = self._p1 = e.pos()
        elif e.button() == Qt.RightButton:
            for items in (self.boxes, self.preds):
                i = self._hit(items, e.pos())
                if i is not None:
                    items.pop(i)
                    self.changed.emit()
                    break
            self.update()

    def mouseMoveEvent(self, e) -> None:
        if self._p0 is not None:
            self._p1 = e.pos()
            self.update()

    def mouseReleaseEvent(self, e) -> None:
        if e.button() != Qt.LeftButton or self._p0 is None:
            return
        p0, p1 = self._p0, e.pos()
        self._p0 = self._p1 = None
        if (p1 - p0).manhattanLength() < 5:  # click: accept a suggestion
            i = self._hit(self.preds, p1)
            if i is not None:
                b = self.preds.pop(i)
                b.score = 1.0
                self.boxes.append(b)
                self.changed.emit()
        else:
            (x1, y1), (x2, y2) = self._to_img(p0), self._to_img(p1)
            b = Box(self.cur_cls, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            if b.x2 - b.x1 > 2 and b.y2 - b.y1 > 2:
                self.boxes.append(b)
                self.changed.emit()
        self.update()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Few-shot labeler")
        self.engine = None
        self.images: list[Path] = []
        self.idx = -1
        self.ann: dict[Path, list[Box]] = {}
        self.sizes: dict[Path, tuple[int, int]] = {}
        self.classes: list[str] = []
        self.labels_dir: Path | None = None

        self.canvas = Canvas()
        self.canvas.changed.connect(self.on_changed)
        self.files = QListWidget()
        self.files.setFixedWidth(220)
        self.files.currentRowChanged.connect(self.show_image)
        self.cls_list = QListWidget()
        self.cls_list.currentRowChanged.connect(self._set_class)
        self.thr_label = QLabel()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(30, 95)
        self.slider.setValue(60)
        self.slider.valueChanged.connect(self.on_threshold)
        self.auto = QCheckBox("Auto-suggest")
        self.auto.setChecked(True)

        side = QVBoxLayout()
        side.addWidget(QLabel("Classes (keys 1-9)"))
        side.addWidget(self.cls_list)
        for text, fn in (("Add class", self.add_class), ("Suggest (S)", self.suggest),
                         ("Accept all (Enter)", self.accept_all), ("Save (Ctrl+S)", self.save),
                         ("Open folder", self.open_folder)):
            btn = QPushButton(text)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.clicked.connect(fn)
            side.addWidget(btn)
        side.addWidget(self.thr_label)
        side.addWidget(self.slider)
        side.addWidget(self.auto)
        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setFixedWidth(200)

        root = QHBoxLayout()
        root.addWidget(self.files)
        root.addWidget(self.canvas, 1)
        root.addWidget(side_w)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)
        for w in (self.files, self.cls_list, self.slider, self.auto):
            w.setFocusPolicy(Qt.NoFocus)
        self.on_threshold()

        def sc(key: str, fn) -> None:
            QShortcut(QKeySequence(key), self, activated=fn)

        sc("Right", lambda: self.step(1))
        sc("Left", lambda: self.step(-1))
        sc("S", self.suggest)
        sc("Return", self.accept_all)
        sc("Ctrl+S", self.save)
        for i in range(1, 10):
            sc(str(i), lambda i=i: self.cls_list.setCurrentRow(i - 1))

    # ---- folder / IO -----------------------------------------------------
    def open_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Image folder")
        if not d:
            return
        root = Path(d)
        self.images = sorted(p for p in root.iterdir() if p.suffix.lower() in IMG_EXT)
        self.labels_dir = root / "labels"
        self.ann, self.sizes = {}, {}
        cf = self.labels_dir / "classes.txt"
        self.classes = cf.read_text(encoding="utf-8").split() if cf.exists() else []
        for p in self.images:
            sz = QImageReader(str(p)).size()
            self.sizes[p] = (sz.width(), sz.height())
            lf = self.labels_dir / f"{p.stem}.txt"
            if lf.exists():
                self.ann[p] = self._read_yolo(lf, *self.sizes[p])
        self.files.blockSignals(True)
        self.files.clear()
        self.files.addItems([p.name for p in self.images])
        self.files.blockSignals(False)
        self._refresh_classes()
        log.info("open_folder path=%s images=%d classes=%d", root, len(self.images), len(self.classes))
        if not self.classes:
            self.add_class()
        if self.images:
            self.files.setCurrentRow(0)

    @staticmethod
    def _read_yolo(lf: Path, w: int, h: int) -> list[Box]:
        boxes = []
        for line in lf.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:  # skip polygon rows
                continue
            c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
            boxes.append(Box(c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
        return boxes

    def save(self) -> None:
        if self.labels_dir is None:
            return
        self.labels_dir.mkdir(exist_ok=True)
        (self.labels_dir / "classes.txt").write_text("\n".join(self.classes) + "\n", encoding="utf-8")
        for p, boxes in self.ann.items():
            w, h = self.sizes[p]
            lines = [f"{b.cls} {(b.x1 + b.x2) / 2 / w:.6f} {(b.y1 + b.y2) / 2 / h:.6f} "
                     f"{(b.x2 - b.x1) / w:.6f} {(b.y2 - b.y1) / h:.6f}" for b in boxes]
            (self.labels_dir / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        log.info("save labels=%d dir=%s", len(self.ann), self.labels_dir)
        self.statusBar().showMessage(f"Saved {len(self.ann)} label files", 3000)

    # ---- classes -----------------------------------------------------------
    def add_class(self) -> None:
        name, ok = QInputDialog.getText(self, "New class", "Class name:")
        if ok and name.strip():
            self.classes.append(name.strip().replace(" ", "_"))
            self._refresh_classes()
            self.cls_list.setCurrentRow(len(self.classes) - 1)

    def _refresh_classes(self) -> None:
        self.cls_list.blockSignals(True)
        self.cls_list.clear()
        self.cls_list.addItems([f"{i + 1}: {n}" for i, n in enumerate(self.classes)])
        self.cls_list.blockSignals(False)
        self.canvas.names = self.classes
        if self.classes:
            self.cls_list.setCurrentRow(max(0, self.cls_list.currentRow()))

    def _set_class(self, row: int) -> None:
        if row >= 0:
            self.canvas.cur_cls = row

    # ---- navigation / suggestions -----------------------------------------
    def step(self, d: int) -> None:
        if self.images:
            self.files.setCurrentRow(min(max(self.idx + d, 0), len(self.images) - 1))

    def show_image(self, i: int) -> None:
        if i < 0:
            return
        self.idx = i
        path = self.images[i]
        self.canvas.set_image(QPixmap(str(path)), self.ann.setdefault(path, []))
        self.statusBar().showMessage(f"{path.name}  ({i + 1}/{len(self.images)})")
        if self.auto.isChecked():
            self.suggest(quiet=True)

    def on_changed(self) -> None:
        if self.auto.isChecked():
            self.suggest(quiet=True)
        self.canvas.update()

    def on_threshold(self) -> None:
        self.thr_label.setText(f"Similarity threshold: {self.slider.value() / 100:.2f}")
        if self.idx >= 0 and self.auto.isChecked():
            self.suggest(quiet=True)

    def suggest(self, quiet: bool = False) -> None:
        if self.idx < 0:
            return
        if not any(self.ann.values()):
            if not quiet:
                self.statusBar().showMessage("Draw a few example boxes first")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if self.engine is None:
                from few_shot import (
                    FewShotEngine,  # loads torch, takes a few seconds once
                )
                self.engine = FewShotEngine()
            self.canvas.preds = self.engine.suggest(self.images[self.idx], self.ann, self.slider.value() / 100)
        finally:
            QApplication.restoreOverrideCursor()
        self.canvas.update()

    def accept_all(self) -> None:
        for b in self.canvas.preds:
            b.score = 1.0
            self.canvas.boxes.append(b)
        self.canvas.preds = []
        self.on_changed()

    def closeEvent(self, e) -> None:
        self.save()
        e.accept()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s level=%(levelname)s logger=%(name)s %(message)s")
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1400, 850)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()