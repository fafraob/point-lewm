"""Point-cloud encoders for the JEPA world models (Point-LeWM / Point-Delta-JEPA).

See :mod:`pc_encoders.base` for the encoder interface and the packed-batch
format, and :mod:`pc_encoders.collate` for the DataLoader ``collate_fn`` that
produces that packed batch. Encoder shipped: :mod:`~pc_encoders.pointvit_encoder`
(Point-BERT-style tokenizer + the image ViT). Shared pieces:
:mod:`~pc_encoders.sampling` (deterministic GPU FPS).
"""

from .base import PackedPointCloud, PointCloudEncoder
from .collate import collate_point_cloud

__all__ = [
    "PointCloudEncoder",
    "PackedPointCloud",
    "collate_point_cloud",
]
