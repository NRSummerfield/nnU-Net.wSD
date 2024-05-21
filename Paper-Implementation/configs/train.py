from typing import Iterable

import argparse, logging, os, torch, warnings

class Config:
    """
    Configurations for a single monai module
    
    - Properties:
        - batch_size: An `int` of batch size
        - data: A `str` of dataset directory
        - device: A `torch.device` to be trained on
        - epochs: An `int` of total training epochs
        - experiment: A `str` of experiment name
        - experiment_dir: A `str` of experiment directory
        - img_size: A tuple of the image size in `int`
        - output_model: A `str` of output model path
        - show_verbose: A `bool` flag of if showing training progress bar
        - use_multi_gpus: A `bool` flag of if using multi GPUs
    """
    batch_size: int
    data: str
    device: torch.device
    epochs: int
    experiment: str
    img_size: tuple[int, ...]
    output_model: str
    show_verbose: bool
    use_multi_gpus: bool
    
    @property
    def experiment_dir(self) -> str:
        return os.path.join("experiments", self.experiment)

    def __init__(
        self, 
        data: str, 
        output_model: str, 
        batch_size: int = 1, 
        device: str = "cuda", 
        epochs: int = 600, 
        experiment: str = "test.exp", 
        img_size: Iterable[int] = (96, 96, 96),
        training_split: int = 4, 
        show_verbose: bool = False, 
        use_multi_gpus: bool = False
        ) -> None:

        """Constructor"""
        # initialize parameters
        super().__init__()
        self.batch_size = batch_size
        self.data = os.path.normpath(data)
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.epochs = epochs
        self.experiment = experiment if experiment.endswith(".exp") else f'{experiment}.exp'
        self.img_size = tuple(img_size)
        if len(self.img_size)==1: self.img_size=(self.img_size[0],self.img_size[0],self.img_size[0])
        self.training_split = training_split
        self.output_model = os.path.normpath(output_model)
        self.show_verbose = show_verbose
        self.use_multi_gpus = use_multi_gpus

        # initialize log
        os.makedirs(self.experiment_dir, exist_ok=True)
        log_path = os.path.join(self.experiment_dir, self.experiment.replace(".exp", ".log"))
        logging.basicConfig(level=logging.INFO, filename=log_path, format="%(message)s")
        warnings.filterwarnings("ignore")
        if self.show_verbose:
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            formatter = logging.Formatter('%(message)s')
            console.setFormatter(formatter)
            logging.getLogger().addHandler(console)

        # assert properties
        assert self.batch_size > 0, f"[Argument Error]: Batch size must be positive, got {self.batch_size}."
        assert self.epochs > 0, f"[Argument Error]: Epochs must be positive, got {self.epochs}."
        for s in self.img_size: assert s > 0, f"[Argument Error]: Image size must be positive numbers, got {self.img_size}."

    @classmethod
    def from_arguments(cls, parser: argparse.ArgumentParser = argparse.ArgumentParser()):
        # add required arguments
        parser.add_argument("data", type=str, help="Data directory.")
        parser.add_argument("output_model", type=str, help="Directory for output model.")
        
        # add training arguments
        training_group = parser.add_argument_group("Training Arguments")
        training_group.add_argument("-b", "--batch_size", type=int, default=1, help="Training batch size, default is 1.")
        training_group.add_argument("--device", type=str, default="cuda", help="The device that training with, default is 'cuda'.")
        training_group.add_argument("-e", "--epochs", type=int, default=600, help="Training epochs, default is 600.")
        training_group.add_argument("-exp", "--experiment", type=str, default="test.exp", help="Name of the experiment, default is 'test.exp'.")
        training_group.add_argument("--img_size", type=int, nargs="+", default=[96, 96, 96], help="The image size, default is 96.")
        training_group.add_argument("-ts", "--training_split", type=int, default=4, help="The index to split the training data, creating a validation set, default is 4")
        training_group.add_argument("--show_verbose", action="store_true", default=False, help="Flag to show progress bar during training.")
        training_group.add_argument("--use_multi_gpus", action="store_true", default=False, help="Flag to use multi GPUs during training.")

        # parsing arguments
        arguments = parser.parse_args().__dict__
        return cls(**arguments)

    def show_settings(self) -> None:
        logging.info(f"Experiment {self.experiment}: output_model_path={self.output_model}")
        logging.info(f"Dataset: path={self.data}")
        logging.info(f"Training settings: epochs={self.epochs}, batch_size={self.batch_size}, img_size={self.img_size}")
        logging.info(f"View settings: show_verbose={self.show_verbose}")
        logging.info(f"Device settings: device={self.device}, use_multi_gpus={self.use_multi_gpus}")
        logging.info("---------------------------------------")
