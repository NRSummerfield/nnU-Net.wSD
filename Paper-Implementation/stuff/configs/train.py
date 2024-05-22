import argparse, os, torchmanager, warnings
from torchmanager_core.view import logger, logging
from typing import Optional, Union

from .basic import Config as _Config

class Config(_Config):
    """
    Configurations for a single monai module
    
    - Properties:
        - experiment: A `str` of experiment name
    """
    experiment: str
    
    @property
    def experiment_dir(self) -> str:
        return os.path.join("experiments", self.experiment)

    def __init__(
        self, 
        device: str = "cuda",
        experiment: str = "test.exp", 
        img_size: list[int] = [96, 96, 64],
        show_verbose: bool = False,
        use_multi_gpus: bool = False,
        overwrite: bool = False
        ) -> None:

        """Constructor"""
        # initialize parameters
        super().__init__(device=device, show_verbose=show_verbose, use_multi_gpus=use_multi_gpus, overwrite=overwrite)
        self.experiment = experiment if experiment.endswith(".exp") else f'{experiment}.exp'
        if len(img_size) == 3: self.img_size = tuple(img_size)
        elif len(img_size) == 1: self.img_size = (img_size[0], img_size[0], img_size[0])
        else: raise ValueError(f'Image size must be in 3-dimension or 1-dimension, got length {len(img_size)}')

        # initialize log path
        os.makedirs(self.experiment_dir, exist_ok=True)
        log_file = os.path.basename(self.experiment.replace(".exp", ".log"))
        log_path = os.path.join(self.experiment_dir, log_file)

        # initialize logger
        logging.getLogger().handlers.clear()
        formatter = logging.Formatter("%(message)s")
        logger.setLevel(logging.INFO)

        # initilaize file handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        warnings.filterwarnings("ignore")

        # initialize console handler
        if self.show_verbose:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # assert properties
        assert torchmanager.version >= "1.1.0", f"Version v1.1.0+ is required for torchmanager, got {torchmanager.version}."
        for s in self.img_size: assert s > 0, f"Image size must be positive numbers, got {self.img_size}."

    def _show_settings(self) -> None:
        logger.info(f"Experiment {self.experiment}")

    @staticmethod
    def get_parser(parser: argparse.ArgumentParser = argparse.ArgumentParser()) -> argparse.ArgumentParser:        
        # add training arguments
        parser = _Config.get_parser(parser)
        training_group = parser.add_argument_group("Training Arguments")
        training_group.add_argument("-exp", "--experiment", type=str, default="test.exp", help="Name of the experiment, default is 'test.exp'.")
        return parser
