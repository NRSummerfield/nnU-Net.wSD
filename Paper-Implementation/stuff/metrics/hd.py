from typing import Any, Optional

from monai.metrics.hausdorff_distance import HausdorffDistanceMetric as _HD

from torchmanager_monai.metrics import CumulativeIterationMetric as Metric

class HausdorffDistanceMetric(Metric):
    """
    The monai `HausdorffDistance` with torchmanager wrap that calculates HD along dimension

    * extends: `torchmanager_monai.metrics.CumulativeIterationMetric`
    """
    _metric_fn: _HD

    def __init__(self, *args: Any, dim: int = 0, target: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(_HD(*args, reduction='none', **kwargs), dim=dim, target=target)