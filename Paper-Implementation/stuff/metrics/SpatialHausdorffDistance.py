from monai.metrics import HausdorffDistanceMetric
from monai.utils import MetricReduction
from monai.config import TensorOrList
from monai.transforms import Resize
import numpy as np
import torch

from typing import Optional, Union

class SpatialHausdorffDistance(HausdorffDistanceMetric):
    def __init__(self, 
            src_dim: Union[list[float], float] = 1,
            src_size: Union[list[int], int] = 1,
            dst_dim: Union[list[float], float] = 1,
            ndim:int = 3,

            include_background: bool = False,
            distance_metric: str = "euclidean",
            percentile: Optional[float] = None, 
            directed: bool = False, 
            reduction: Union[MetricReduction, str] = MetricReduction.MEAN, 
            get_not_nans: bool = False
            ) -> None:
        

        super().__init__(include_background, distance_metric, percentile, directed, reduction, get_not_nans)
        dim_change =  np.asarray(src_dim) / np.asarray(dst_dim)
        new_size = np.array(np.asarray(src_size) * dim_change, dtype=int)
        self.s = list(new_size)
        # new_size = [nchannels] + list(new_size)
        new_size = list(new_size)
        self.resize = Resize(spatial_size=new_size, mode='nearest')

    
    def __call__(self, y_pred: TensorOrList, y: Optional[TensorOrList] = None):  # type: ignore
        # batch, channel, X, Y, Z
        out = []
        b, c, *dim = y.shape
        new_size = [b, c] + self.s

        for tensor in [y_pred, y]:
            new_tensor = torch.ones(new_size)
            
            for b in range(tensor.shape[0]):
                new_tensor[b] = self.resize(tensor[b])

            out.append(new_tensor)
        
        ret = super().__call__(y_pred=out[0], y=out[1])
        return ret

if __name__ == '__main__':

    from torchmanager_monai import metrics
    sHD = metrics.CumulativeIterationMetric(SpatialHausdorffDistance(
        src_dim=[1, 1, 2],
        src_size=[128, 128, 64],
        dst_dim=[1, 1, 1],
        include_background=True, 
        reduction="none", 
        get_not_nans=False, 
        percentile=95
    ))

    y_pred = torch.randint(low=0, high=2, size=(1, 13, 128, 128, 64))
    y = torch.randint(low=0, high=2, size=(1, 13, 128, 128, 64))

    sHD(y_pred, y)
    print(sHD.result)
    print(sHD.sub_results)

    # print(torch.ones([1, 13, 128, 128, 128]).shape)

