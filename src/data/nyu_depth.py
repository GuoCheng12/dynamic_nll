from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from torchvision import transforms
    from torchvision.transforms import functional as TF
except ImportError as exc:  # pragma: no cover
    raise ImportError("torchvision is required for NYUDepthDataset") from exc


class NYUDepthDataset(Dataset):
    """
    NYUv2 depth dataset loader with optional dummy data mode.
    Expected filenames file format: <rgb_path> <depth_path> [focal]
    """

    def __init__(
        self,
        data_path: str,
        gt_path: str,
        filenames_file: str,
        input_size: Tuple[int, int] = (256, 256),
        mode: str = "train",
        use_dummy_data: bool = False,
        n_samples: int = 100,
        do_random_rotate: bool = False,
        degree: float = 2.5,
    ):
        self.data_path = data_path
        self.gt_path = gt_path
        self.filenames_file = filenames_file
        self.input_size = input_size
        self.mode = mode
        self.use_dummy_data = use_dummy_data
        self.n_samples = n_samples
        self.do_random_rotate = do_random_rotate
        self.degree = degree
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if not use_dummy_data:
            with open(filenames_file, "r") as f:
                self.lines = [line.strip() for line in f if line.strip()]
        else:
            self.lines = []

    def __len__(self) -> int:
        return self.n_samples if self.use_dummy_data else len(self.lines)

    def _load_paths(self, line: str) -> Tuple[str, str]:
        parts = line.split()
        image_path = os.path.join(self.data_path, parts[0])
        depth_path = os.path.join(self.gt_path, parts[1])
        return image_path, depth_path

    def __getitem__(self, idx):
        if self.use_dummy_data:
            c, h, w = 3, self.input_size[0], self.input_size[1]
            image = torch.rand(c, h, w)
            depth = torch.rand(1, h, w)
            return image, depth

        image_path, depth_path = self._load_paths(self.lines[idx])
        image = Image.open(image_path).convert("RGB")
        depth = Image.open(depth_path)

        if self.mode == "train" and self.do_random_rotate:
            angle = (torch.rand(1).item() - 0.5) * 2 * self.degree
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            depth = TF.rotate(depth, angle, interpolation=TF.InterpolationMode.NEAREST)

        image = TF.resize(image, self.input_size, interpolation=TF.InterpolationMode.BILINEAR)
        depth = TF.resize(depth, self.input_size, interpolation=TF.InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        image = self.normalize(image)

        depth_np = np.asarray(depth, dtype=np.float32)
        depth_np = depth_np / 1000.0
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)

        return image, depth_tensor
