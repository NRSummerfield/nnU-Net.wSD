import torch
from torch.utils.data import DataLoader as _TorchDataLoader
from torch.utils.data import Dataset
from monai.data.utils import list_data_collate, set_rnd, worker_init_fn


import collections.abc
import sys
import warnings
from copy import copy, deepcopy
from multiprocessing.pool import ThreadPool
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset as _TorchDataset
from torch.utils.data import Subset

from monai.data.utils import pickle_hashing
from monai.transforms import Compose, Randomizable, ThreadUnsafe, Transform, apply_transform, convert_to_contiguous
from monai.utils import min_version, optional_import

if TYPE_CHECKING:
    from tqdm import tqdm

    has_tqdm = True
else:
    tqdm, has_tqdm = optional_import("tqdm", "4.47.0", min_version, "tqdm")

lmdb, _ = optional_import("lmdb")
pd, _ = optional_import("pandas")



__all__ = ["DataLoader"]


class DataLoader(_TorchDataLoader):
    def __init__(self, dataset: Dataset, num_workers: int = 0, **kwargs) -> None:
        if num_workers == 0:
            # when num_workers > 0, random states are determined by worker_init_fn
            # this is to make the behavior consistent when num_workers == 0
            # torch.int64 doesn't work well on some versions of windows
            _seed = torch.empty((), dtype=torch.int32).random_(generator=None).item()
            set_rnd(dataset, int(_seed))
        if "collate_fn" not in kwargs:
            kwargs.update({"collate_fn": list_data_collate})
        if "worker_init_fn" not in kwargs:
            kwargs.update({"worker_init_fn": worker_init_fn})

        super().__init__(dataset=dataset, num_workers=num_workers, **kwargs)


class Dataset(_TorchDataset):
    """
    A generic dataset with a length property and an optional callable data transform
    when fetching a data sample.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    For example, typical input data can be a list of dictionaries::

        [{                            {                            {
             'img': 'image1.nii.gz',      'img': 'image2.nii.gz',      'img': 'image3.nii.gz',
             'seg': 'label1.nii.gz',      'seg': 'label2.nii.gz',      'seg': 'label3.nii.gz',
             'extra': 123                 'extra': 456                 'extra': 789
         },                           },                           }]
    """

    def __init__(self, data: Sequence, transform: Optional[Callable] = None) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: a callable data transform on input data.

        """
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        data_i = self.data[index]
        return apply_transform(self.transform, data_i) if self.transform is not None else data_i

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # dataset[[1, 3, 4]]
            return Subset(dataset=self, indices=index)
        return self._transform(index)
    

class CacheDataset(Dataset):
    def __init__(
        self,
        data: Sequence,
        transform: Optional[Union[Sequence[Callable], Callable]] = None,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: Optional[int] = 1,
        progress: bool = True,
        copy_cache: bool = True,
        as_contiguous: bool = True,
        hash_as_key: bool = False,
        hash_func: Callable[..., bytes] = pickle_hashing,
    ) -> None:
        if not isinstance(transform, Compose):
            transform = Compose(transform)
        super().__init__(data=data, transform=transform)
        self.set_num = cache_num  # tracking the user-provided `cache_num` option
        self.set_rate = cache_rate  # tracking the user-provided `cache_rate` option
        self.progress = progress
        self.copy_cache = copy_cache
        self.as_contiguous = as_contiguous
        self.hash_as_key = hash_as_key
        self.hash_func = hash_func
        self.num_workers = num_workers
        if self.num_workers is not None:
            self.num_workers = max(int(self.num_workers), 1)
        self.cache_num = 0
        self._cache: Union[List, Dict] = []
        self.set_data(data)

    def set_data(self, data: Sequence):
        """
        Set the input data and run deterministic transforms to generate cache content.

        Note: should call this func after an entire epoch and must set `persistent_workers=False`
        in PyTorch DataLoader, because it needs to create new worker processes based on new
        generated cache content.

        """

        def _compute_cache():
            self.cache_num = min(int(self.set_num), int(len(self.data) * self.set_rate), len(self.data))
            return self._fill_cache()

        if self.hash_as_key:
            # only compute cache for the unique items of dataset
            mapping = {self.hash_func(v): v for v in data}
            self.data = list(mapping.values())
            cache_ = _compute_cache()
            self._cache = dict(zip(list(mapping)[: self.cache_num], cache_))
            self.data = data
        else:
            self.data = data
            self._cache = _compute_cache()

    def _fill_cache(self) -> List:
        if self.cache_num <= 0:
            return []
        if self.progress and not has_tqdm:
            warnings.warn("tqdm is not installed, will not show the caching progress bar.")
        with ThreadPool(self.num_workers) as p:
            if self.progress and has_tqdm:
                return list(
                    tqdm(
                        p.imap(self._load_cache_item, range(self.cache_num)),
                        total=self.cache_num,
                        desc="Loading dataset",
                    )
                )
            return list(p.imap(self._load_cache_item, range(self.cache_num)))

    def _load_cache_item(self, idx: int):
        """
        Args:
            idx: the index of the input data sequence.
        """
        item = self.data[idx]
        for _transform in self.transform.transforms:  # type:ignore
            # execute all the deterministic transforms
            if isinstance(_transform, Randomizable) or not isinstance(_transform, Transform):
                break
            _xform = deepcopy(_transform) if isinstance(_transform, ThreadUnsafe) else _transform
            item = apply_transform(_xform, item)
        if self.as_contiguous:
            item = convert_to_contiguous(item, memory_format=torch.contiguous_format)
        return item

    def _transform(self, index: int):
        index_: Any = index
        if self.hash_as_key:
            key = self.hash_func(self.data[index])
            if key in self._cache:
                # if existing in cache, get the index
                index_ = key  # if using hash as cache keys, set the key

        if isinstance(index_, int) and index_ % len(self) >= self.cache_num:  # support negative index
            # no cache for this index, execute all the transforms directly
            return super()._transform(index_)
        # load data from cache and execute from the first random transform
        start_run = False
        if self._cache is None:
            self._cache = self._fill_cache()
        data = self._cache[index_]
        if not isinstance(self.transform, Compose):
            raise ValueError("transform must be an instance of monai.transforms.Compose.")
        for _transform in self.transform.transforms:
            if start_run or isinstance(_transform, Randomizable) or not isinstance(_transform, Transform):
                # only need to deep copy data on first non-deterministic transform
                if not start_run:
                    start_run = True
                    if self.copy_cache:
                        data = deepcopy(data)
                data = apply_transform(_transform, data)
        return data