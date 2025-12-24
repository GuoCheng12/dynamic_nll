from .base_dataset import ToyRegressionDataset, build_dataloaders
from .heteroscedastic_sine import PaperSineDataset
from .nyu_depth import NYUDepthDataset, build_nyu_dataloaders

__all__ = ["ToyRegressionDataset", "build_dataloaders", "PaperSineDataset", "NYUDepthDataset", "build_nyu_dataloaders"]
