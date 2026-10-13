"""Strict labelme parser.

Facts below were verified on the real training data (2092 label files) and are
enforced here; anything else raises :class:`LabelmeError` instead of being
silently tolerated.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ALLOWED_LABELS: dict[str, frozenset[str]] = {
    "EO": frozenset({"solar_panel", "crack"}),
    "IR": frozenset({"solar_panel", "hotspot"}),
}
DEFECT_LABEL: dict[str, str] = {"EO": "crack", "IR": "hotspot"}
IMAGE_SIZE: dict[str, tuple[int, int]] = {"EO": (1920, 1080), "IR": (640, 512)}  # (width, height)
PANEL = "solar_panel"
BOUNDS_TOL = 1e-6  # observed overshoot of coordinates is ~5e-13 px


class LabelmeError(ValueError):
    """Label file does not match the verified dataset structure."""


@dataclass(frozen=True)
class Shape:
    label: str
    points: tuple[tuple[float, float], ...]
    reason: str
    auto_label: bool
    description: str


@dataclass(frozen=True)
class LabelmeDoc:
    version: str
    image_path: str
    width: int
    height: int
    modality: str
    shapes: tuple[Shape, ...]

    @property
    def panels(self) -> tuple[Shape, ...]:
        return tuple(s for s in self.shapes if s.label == PANEL)

    @property
    def defects(self) -> tuple[Shape, ...]:
        return tuple(s for s in self.shapes if s.label != PANEL)

    @property
    def n_panels(self) -> int:
        return len(self.panels)

    @property
    def n_defects(self) -> int:
        return len(self.defects)

    @property
    def n_auto_panels(self) -> int:
        return sum(1 for s in self.shapes if s.auto_label and s.label == PANEL)

    @property
    def has_auto_label(self) -> bool:
        return any(s.auto_label for s in self.shapes)


def _err(src: str, msg: str) -> LabelmeError:
    return LabelmeError(f"{src}: {msg}")


def _parse_points(raw: Any, width: int, height: int, src: str, idx: int) -> tuple[tuple[float, float], ...]:
    if not isinstance(raw, list) or len(raw) < 3:
        n = len(raw) if isinstance(raw, list) else "n/a"
        raise _err(src, f"shape {idx}: polygon needs at least 3 points, got {n}")
    out: list[tuple[float, float]] = []
    for p in raw:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise _err(src, f"shape {idx}: point must be [x, y], got {p!r}")
        x, y = p
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (x, y)):
            raise _err(src, f"shape {idx}: non-numeric point {p!r}")
        if not (math.isfinite(x) and math.isfinite(y)):
            raise _err(src, f"shape {idx}: non-finite point {p!r}")
        if x < -BOUNDS_TOL or y < -BOUNDS_TOL or x > width + BOUNDS_TOL or y > height + BOUNDS_TOL:
            raise _err(src, f"shape {idx}: point {p!r} outside the {width}x{height} image")
        out.append((float(x), float(y)))
    return tuple(out)


def parse_labelme(
    data: Mapping[str, Any],
    modality: str,
    *,
    actual_size: tuple[int, int] | None = None,
    source: str = "<dict>",
) -> LabelmeDoc:
    """Parse a labelme dict. ``actual_size`` is the real image (width, height)."""
    if modality not in ALLOWED_LABELS:
        raise ValueError(f"unknown modality {modality!r}")
    for key in ("version", "shapes", "imagePath", "imageWidth", "imageHeight", "imageData"):
        if key not in data:
            raise _err(source, f"missing top-level key {key!r}")
    if data["imageData"] is not None:
        raise _err(source, "imageData must be null")
    if not isinstance(data["version"], str):
        raise _err(source, f"version must be a string, got {data['version']!r}")
    w, h = data["imageWidth"], data["imageHeight"]
    if isinstance(w, bool) or isinstance(h, bool) or not isinstance(w, int) or not isinstance(h, int):
        raise _err(source, f"imageWidth/imageHeight must be integers, got {w!r}/{h!r}")
    if (w, h) != IMAGE_SIZE[modality]:
        raise _err(source, f"{modality} label declares {w}x{h}, expected {IMAGE_SIZE[modality][0]}x{IMAGE_SIZE[modality][1]}")
    if actual_size is not None and tuple(actual_size) != (w, h):
        raise _err(source, f"label declares {w}x{h} but the image is {actual_size[0]}x{actual_size[1]}")
    if not isinstance(data["shapes"], list):
        raise _err(source, "shapes must be a list")

    allowed = ALLOWED_LABELS[modality]
    shapes: list[Shape] = []
    for i, s in enumerate(data["shapes"]):
        if not isinstance(s, Mapping):
            raise _err(source, f"shape {i}: must be an object")
        label = s.get("label")
        if label not in allowed:
            raise _err(source, f"shape {i}: label {label!r} not allowed for {modality} (allowed: {sorted(allowed)})")
        if s.get("shape_type") != "polygon":
            raise _err(source, f"shape {i}: shape_type must be 'polygon', got {s.get('shape_type')!r}")
        if "reason" not in s:
            raise _err(source, f"shape {i}: missing 'reason' key")
        reason = s["reason"]
        if label == PANEL:  # verified: reason is JSON null on every solar_panel
            if reason is not None:
                raise _err(source, f"shape {i}: solar_panel reason must be null, got {reason!r}")
            reason = ""
        elif not isinstance(reason, str) or not reason.strip():
            raise _err(source, f"shape {i}: {label} must have a non-empty string reason, got {reason!r}")
        flags = s.get("flags", {})
        if not isinstance(flags, Mapping):
            raise _err(source, f"shape {i}: flags must be an object")
        desc = s.get("description")
        shapes.append(
            Shape(
                label=label,
                points=_parse_points(s.get("points"), w, h, source, i),
                reason=reason,
                auto_label=flags.get("auto_label") is True,
                description=desc if isinstance(desc, str) else "",
            )
        )
    return LabelmeDoc(
        version=data["version"], image_path=str(data["imagePath"]), width=w, height=h,
        modality=modality, shapes=tuple(shapes),
    )


def load_labelme(path: str | Path, modality: str, *, actual_size: tuple[int, int] | None = None) -> LabelmeDoc:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise _err(str(path), "top level must be an object")
    return parse_labelme(data, modality, actual_size=actual_size, source=str(path))
