from monai.data import * # type: ignore

from . import datasets, transforms
from .challenge import load as load_challenge
from .general import TransformOptions, load #, load_SingleVolume
from .ma_general import load_SingleVolume, load_SingleVolume_advanced

from .viewray import load as load_ViewRay

from .ctca_coronaryArteries import load as load_CTCA_CA

try: from .iseg2017 import load as load_iseg2017
except ImportError: load_iseg2017 = NotImplemented


from .data_manager import DataLoader, Dataset, CacheDataset

VERSION = 'v0.2.0b'
