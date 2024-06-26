from __future__ import annotations
from typing import Any, Callable, Generic, Sequence, Type, Union, Optional

import torch, torchmanager as tm
from monai.data.utils import decollate_batch
from monai.inferers.utils import sliding_window_inference
from torchmanager_core import devices, view
from torchmanager_core.typing import Module, SizedIterable

from .protocols import SubResulting

class Manager(tm.Manager, Generic[Module]):
    """The training manager with monai compability"""
    _post_labels: Callable[[torch.Tensor], torch.Tensor]
    _post_predicts: Callable[[torch.Tensor], torch.Tensor]
    _roi_size: Sequence[int]
    model: Module

    def __init__(self, *args: Any, post_labels: Callable[[torch.Tensor], torch.Tensor] = lambda x: x, post_predicts: Callable[[torch.Tensor], torch.Tensor] = lambda x: x, roi_size: Sequence[int] = (96, 96, 96), **kwargs: Any) -> None:
        """
        Constructor

        - Parameters:
            - post_labels: A `list` of post `Callable[[torch.Tensor], torch.Tensor]`s for labels
            - post_predicts: A `list` of post `Callable[[torch.Tensor], torch.Tensor]`s for predictions
            - roi_size: A `Sequence` of sliding window size in `int` during the testing
        """
        super().__init__(*args, **kwargs)
        self._post_labels = post_labels
        self._post_predicts = post_predicts
        self._roi_size = roi_size
        self.notes = ""

    @classmethod
    def from_checkpoint(cls: Type[Manager], *args, **kwargs) -> Manager:
        return super().from_checkpoint(*args, **kwargs) # type: ignore

    @torch.no_grad()
    def predict(self, dataset: SizedIterable, device: Optional[Union[torch.device, list[torch.device]]] = None, use_multi_gpus: bool = False, show_verbose: bool = False) -> list[torch.Tensor]:
        # find available device
        cpu, device, target_devices = devices.search(None if use_multi_gpus else device)
        if device == cpu and len(target_devices) < 2: use_multi_gpus = False
        devices.set_default(target_devices[0])

        # move model
        if use_multi_gpus and not isinstance(self.model, torch.nn.parallel.DataParallel):
            raw_model = self.model
            self.model, use_multi_gpus = devices.data_parallel(self.model, devices=target_devices)
        else: raw_model = None

        # move to device
        self.to(device)

        # initialize predictions
        self.model.eval()
        predictions: list[torch.Tensor] = []
        if len(dataset) == 0: return predictions
        progress_bar = view.tqdm(total=len(dataset)) if show_verbose else None

        # loop the dataset
        for data in dataset:
            x, _ = self.unpack_data(data)
            if use_multi_gpus is not True: x = x.to(device)
            val_outputs = sliding_window_inference(x, self._roi_size, 1, self.model)
            val_outputs_list: list[torch.Tensor] = decollate_batch(val_outputs) # type: ignore
            y_post = [self._post_predicts(val_pred_tensor).unsqueeze(0) for val_pred_tensor in val_outputs_list]
            y = torch.cat(y_post)
            predictions.append(y)
            if progress_bar is not None: progress_bar.update()

        # reset model and loss
        if raw_model is not None: self.model = raw_model.to(cpu)
        devices.empty_cache()
        return predictions

    @torch.no_grad()
    def test(self, dataset: Any, device: Optional[torch.device] = None, use_multi_gpus: bool = False, show_verbose: bool = False, **kwargs: Any) -> dict[str, float]:
        # initialize
        # loss_fn = self.loss_fn # remove to calculate val loss
        # self.loss_fn = None # remove to calculate val loss
        summary = super().test(dataset, device=device, use_multi_gpus=use_multi_gpus, show_verbose=show_verbose, **kwargs)
        # summary = {k: v for k, v in summary.items() if "loss" not in k} # remove to show val loss

        # get sub results
        for k, m in self.compiled_metrics.items():
            if isinstance(m, SubResulting):
                subresults: list[float] = m.sub_results.reshape([-1]).detach().tolist()
                k = k.replace('val_', '')
                for i, r in enumerate(subresults): summary[f"{k}_{i}"] = r

        # reset loss fn
        # self.loss_fn = loss_fn # remove to calculate val loss
        return summary

    def test_step(self, x_test: Any, y_test: dict[str, torch.Tensor]) -> dict[str, float]:
        # initialize
        summary: dict[str, float] = {}

        # forward pass
        val_outputs = sliding_window_inference(x_test, self._roi_size, 1, self.model)
        val_labels_list: list[torch.Tensor] = decollate_batch(y_test['out']) # type: ignore
        y_test_dict = {"out": torch.cat([l.unsqueeze(0) for l in val_labels_list])}
        val_outputs_list: list[torch.Tensor] = decollate_batch(val_outputs) # type: ignore
        y_dict = {"out": torch.cat([o.unsqueeze(0) for o in val_outputs_list])}
        
        if 'dist_map' in y_test:
            val_distmap_list: list[torch.Tensor] = decollate_batch(y_test['dist_map']) # type: ignore
            y_test_dict['dist_map'] = torch.cat([l.unsqueeze(0) for l in val_distmap_list])
            y_dict['dist_map'] = y_dict['out']
        else: 
            val_distmap_list = None # type:ignore          
                     
        # calculate loss
        if self.loss_fn is not None: summary["loss"] = float(self.loss_fn(y_dict, y_test_dict))

        # post process for metrics evaluation
        y_test_post = [self._post_labels(val_label_tensor).unsqueeze(0) for val_label_tensor in val_labels_list]
        y_post = [self._post_predicts(val_pred_tensor).unsqueeze(0) for val_pred_tensor in val_outputs_list]
        y_test_dict["out"] = torch.cat(y_test_post)
        y_dict["out"] = torch.cat(y_post).to(y_test_dict["out"].device)
        if 'dist_map' in y_test:
            y_dict['dist_map'] = y_dict['out']

        # forward metrics
        for name, fn in self.compiled_metrics.items():
            if name.startswith("val_"): name = name.replace("val_", "")
            try:
                fn(y_dict, y_test_dict)
                summary[name] = float(fn.result.detach())
            except Exception as metric_error:
                runtime_error = RuntimeError(f"Cannot fetch metric '{name}'.")
                raise runtime_error from metric_error
        return summary

    def unpack_data(self, data: dict[str, Any]) -> tuple[Any, Union[dict[str, torch.Tensor], torch.Tensor]]:
        # image, label = data["image"], data["label"]
        image= data["image"]
        if "label" in data:
            label = data["label"]
            data.update({"out": label})
        y = {k: v for k, v in data.items() if torch.is_tensor(v)}
        # return image, y if self.model.training is True else data["label"]
        # return image, y if self.model.training is True else y
        return image, y 

