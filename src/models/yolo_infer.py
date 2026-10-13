"""Raw YOLO-seg inference -> cacheable per-image predictions (needs ultralytics)."""
from __future__ import annotations

import logging
from typing import Any, Callable, Iterable

import numpy as np

logger = logging.getLogger(__name__)

Loader = Callable[[], np.ndarray]  # returns an RGB uint8 (H, W, 3) array


def predict_raw(
    model: Any, items: Iterable[tuple[str, Loader]], class_names: list[str], *, imgsz: int, conf: float,
    max_det: int, batch: int, iou: float = 0.7, half: bool = True, device: int | str = 0,
) -> dict[str, dict[str, Any]]:
    """Run inference; every instance keeps class, confidence and the raw mask contour (pixels, original size)."""
    out: dict[str, dict[str, Any]] = {}
    items = list(items)
    for s in range(0, len(items), batch):
        chunk = items[s:s + batch]
        arrays = [ld() for _, ld in chunk]
        for a in arrays:
            if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
                raise ValueError(f"loader must return RGB uint8 (H, W, 3), got {a.shape} {a.dtype}")
        bgr = [np.ascontiguousarray(a[..., ::-1]) for a in arrays]
        results = model.predict(bgr, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, half=half, device=device,
                                retina_masks=True, verbose=False)
        for (key, _), arr, r in zip(chunk, arrays, results):
            h, w = arr.shape[:2]
            preds = []
            if r.masks is not None and len(r.boxes):
                cls = r.boxes.cls.cpu().numpy().astype(int)
                cf = r.boxes.conf.cpu().numpy()
                for k, xy in enumerate(r.masks.xy):
                    preds.append({"cls": int(cls[k]), "label": class_names[int(cls[k])], "conf": float(cf[k]),
                                  "poly": np.asarray(xy, dtype=np.float64).round(3).tolist()})
            out[key] = {"w": w, "h": h, "preds": preds}
        logger.info("predicted %d/%d", min(s + batch, len(items)), len(items))
    return out
