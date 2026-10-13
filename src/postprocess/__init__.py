"""Generic polygon / mask post-processing (no dependence on the official format)."""
from src.postprocess.polygon import mask_to_polygons, polygon_to_mask
from src.postprocess.validate import ValidationResult, validate_polygon

__all__ = ["mask_to_polygons", "polygon_to_mask", "ValidationResult", "validate_polygon"]
