import abc, argparse, logging, monai, os, torch, torchmanager, warnings

class Config(abc.ABC):
    device: torch.device
    show_verbose: bool
    use_multi_gpus: bool
    overwrite: bool

    def __init__(self, device: str = "cuda", show_verbose: bool = True, use_multi_gpus: bool = False, overwrite: bool = False):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.show_verbose = show_verbose
        self.use_multi_gpus = use_multi_gpus if torch.cuda.is_available() else False
        self.overwrite = overwrite

        # argument check
        if use_multi_gpus and not torch.cuda.is_available(): warnings.warn("No CUDA detected when using multi GPUs!", ResourceWarning)
        if self.device.type == "cuda" and not torch.cuda.is_available(): raise SystemError("CUDA device cannot be found")
        if self.device.type == "cpu" and use_multi_gpus: raise SystemError("Cannot use multi GPUs by specifying CPU as device.")
        assert torchmanager.version >= "1.0.4", f"Version v1.0.4+ is required for torchmanager, got {torchmanager.version}."

    @abc.abstractmethod
    def _show_settings(self) -> None: pass

    @staticmethod
    def get_parser(parser: argparse.ArgumentParser = argparse.ArgumentParser()) -> argparse.ArgumentParser:
        parser.add_argument("--device", type=str, default="cuda", help="The device that running with, default is 'cuda'.")
        parser.add_argument("--show_verbose", action="store_true", default=False, help="Flag to show progress bar during running.")
        parser.add_argument("--use_multi_gpus", action="store_true", default=False, help="Flag to use multi GPUs during running.")
        parser.add_argument("--overwrite", action='store_true', default=False, help='Flag to overwrite a saved file.')
        return parser

    def show_settings(self) -> None:
        self._show_settings()
        logger = logging.getLogger("torchmanager")
        logger.info(f"View settings: show_verbose={self.show_verbose}")
        logger.info(f"Device settings: device={self.device}, use_multi_gpus={self.use_multi_gpus}")
        if self.overwrite: logger.info("Model set to overwrite any save file")
        logger.info(f"Environments: monai={monai.__version__}, torch={torch.__version__}, torchmanager={torchmanager.version}")
        logger.info("---------------------------------------")

    @classmethod
    def from_arguments(cls, parser: argparse.ArgumentParser = argparse.ArgumentParser()):
        parser = cls.get_parser()
        arguments = parser.parse_args().__dict__
        return cls(**arguments)
