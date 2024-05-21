from typing import Any, Optional

from monai.metrics.meandice import DiceMetric as _DiceMetric
from torchmanager_monai.metrics import CumulativeIterationMetric as Metric

class DiceMetric(Metric):
    """
    The monai `DiceMetric` with torchmanager wrap that calculates dice along dimension

    * extends: `torchmanager_monai.metrics.CumulativeIterationMetric`
    """
    _metric_fn: _DiceMetric

    def __init__(self, *args: Any, dim: int = 0, target: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(_DiceMetric(*args, reduction='none', **kwargs), dim=dim, target=target)