from typing import Any, Optional

from monai.metrics.surface_distance import SurfaceDistanceMetric as _MSD

from torchmanager_monai.metrics import CumulativeIterationMetric as Metric

class SurfaceDistanceMetric(Metric):
    """
    The monai `SurfaceDistanceMetric` with torchmanager wrap that calculates mean distance to agreement MDA (MSD) along dimension

    * extends: `torchmanager_monai.metrics.CumulativeIterationMetric`
    """
    _metric_fn: _MSD

    def __init__(self, *args: Any, dim: int = 0, target: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(_MSD(*args, reduction='none', **kwargs), dim=dim, target=target)
