"""encoder package — public re-exports."""
from .arc_to_24bit import (
    encode_grid, arc_to_24bit, encode_task,
    EncoderReport, TaskEncoding,
    PALETTE_LUT,
)

__all__ = [
    "encode_grid", "arc_to_24bit", "encode_task",
    "EncoderReport", "TaskEncoding", "PALETTE_LUT",
]
