from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

try:
    from torchvision import transforms
    from torchvision.transforms import functional as TF
except ImportError as exc:  # pragma: no cover
    raise ImportError("torchvision is required for NYUDepthDataset") from exc
    
class NYUDepthDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        gt_path: str,
        filenames_file: str,
        input_size: Tuple[int, int] = (416, 544), # Default from reference
        mode: str = "train",
        use_dummy_data: bool = False,
        n_samples: int = 100,
        do_random_rotate: bool = True,
        degree: float = 2.5,
        eigen_crop: bool = True,
        train_crop: Optional[bool] = None,
        min_depth: float = 1e-3,
        max_depth: float = 10.0,
        min_depth_eval: float = 1e-3,
        max_depth_eval: float = 10.0,
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
        self.eigen_crop = eigen_crop
        self.train_crop = eigen_crop if train_crop is None else train_crop
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_depth_eval = min_depth_eval
        self.max_depth_eval = max_depth_eval
        
        # Reference normalization
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if not use_dummy_data:
            resolved = self._resolve_filenames_file(filenames_file, data_path)
            if not os.path.exists(resolved):
                raise FileNotFoundError(f"filenames_file not found: {resolved}")
            with open(resolved, "r") as f:
                self.lines = [line.strip() for line in f if line.strip()]
        else:
            self.lines = []

    @staticmethod
    def _resolve_filenames_file(filenames_file: str, base_path: str) -> str:
        if os.path.isabs(filenames_file):
            return filenames_file
        candidate = os.path.join(base_path, filenames_file)
        if os.path.exists(candidate):
            return candidate
        repo_root = Path(__file__).resolve().parents[2]
        repo_candidate = str(repo_root / filenames_file)
        if os.path.exists(repo_candidate):
            return repo_candidate
        return filenames_file

    @staticmethod
    def _strip_leading_slash(path: str) -> str:
        if path.startswith(("/", "\\")):
            return path[1:]
        return path

    def __len__(self) -> int:
        return self.n_samples if self.use_dummy_data else len(self.lines)

    def _load_paths(self, line: str) -> Tuple[str, str]:
        parts = line.split()
        image_rel = self._strip_leading_slash(parts[0])
        depth_rel = self._strip_leading_slash(parts[1])
        image_path = os.path.join(self.data_path, image_rel)
        depth_path = os.path.join(self.gt_path, depth_rel)
        return image_path, depth_path

    def __getitem__(self, idx):
        if self.use_dummy_data:
            # Dummy implementation for testing
            image = torch.zeros(3, self.input_size[0], self.input_size[1])
            depth = torch.zeros(1, self.input_size[0], self.input_size[1])
            return image, depth

        image_path, depth_path = self._load_paths(self.lines[idx])
        # Open images (Reference: Image.open)
        image = Image.open(image_path)
        depth = Image.open(depth_path)

        if self.mode == 'train':
            # 1. Fixed Crop (Remove white borders) - Exact reference logic
            # "To avoid blank boundaries due to pixel registration"
            image = image.crop((43, 45, 608, 472))
            depth = depth.crop((43, 45, 608, 472))

            # 2. Random Rotate
            if self.do_random_rotate:
                random_angle = (random.random() - 0.5) * 2 * self.degree
                image = self.rotate_image(image, random_angle)
                depth = self.rotate_image(depth, random_angle, flag=Image.NEAREST)

            # 3. To Numpy & Normalize (Reference logic)
            image = np.asarray(image, dtype=np.float32) / 255.0
            depth = np.asarray(depth, dtype=np.float32)
            depth = np.expand_dims(depth, axis=2)

            # 4. Depth Unit Scaling (mm to meters)
            depth = depth / 1000.0

            # 5. Random Crop
            image, depth = self.random_crop(image, depth, self.input_size[0], self.input_size[1]) # H, W

            # 6. Train Preprocess (Flip & Augmentations)
            image, depth = self.train_preprocess(image, depth)

            # 7. To Tensor & Normalize
            image_tensor = self.to_tensor(image)
            image_tensor = self.normalize(image_tensor)
            
            depth_tensor = self.to_tensor(depth).float() # Returns 1xHxW

            return image_tensor, depth_tensor

        else:
            # Validation Mode
            image = np.asarray(image, dtype=np.float32) / 255.0
            
            # Validation Load specific (Handling missing depth in online_eval if needed, here we assume existence)
            depth = np.asarray(depth, dtype=np.float32)
            depth = np.expand_dims(depth, axis=2)
            depth = depth / 1000.0

            # KB Crop (optional in ref, usually False for NYU)
            # if self.args.do_kb_crop ... (Ref: mostly for KITTI)
            
            # Ref just returns full image for 'online_eval' usually, or crops to input_size if needed?
            # Ref: "if self.mode == 'online_eval': ... sample = {'image': image ...}"
            # Ref: "if self.mode == 'test': ... image = self.to_tensor(image)"
            
            # We align with our training loop expectation -> Return Tensor
            image_tensor = self.to_tensor(image)
            image_tensor = self.normalize(image_tensor)
            depth_tensor = self.to_tensor(depth).float()

            return image_tensor, depth_tensor

    # --- Reference Helper Methods ---

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        brightness = random.uniform(0.75, 1.25)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def to_tensor(self, pic):
        if not (isinstance(pic, np.ndarray) or isinstance(pic, Image.Image)):
             raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            # Numpy Image: H x W x C -> C x H x W
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img
        
        # PIL (fallback)
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
            
        img = img.view(pic.size[1], pic.size[0], nchannel)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


def build_nyu_dataloaders(
    data_cfg,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    eval_distributed: bool | None = None,
):
    if eval_distributed is None:
        eval_distributed = distributed
    
    # Use Reference defaults if not specified
    input_size = (data_cfg.get("input_height", 416), data_cfg.get("input_width", 544))
    
    batch_size = data_cfg.get("batch_size", 1)
    eval_batch_size = data_cfg.get("eval_batch_size", batch_size)
    num_workers = data_cfg.get("num_workers", 0)

    train_ds = NYUDepthDataset(
        data_path=data_cfg.train_data_path,
        gt_path=data_cfg.train_gt_path,
        filenames_file=data_cfg.filenames_file,
        input_size=input_size,
        mode="train",
        use_dummy_data=data_cfg.use_dummy_data,
        n_samples=data_cfg.n_samples,
        do_random_rotate=data_cfg.get("do_random_rotate", True),
        degree=data_cfg.get("degree", 2.5),
        eigen_crop=data_cfg.get("eigen_crop", True),
        train_crop=data_cfg.get("train_crop", None),
        min_depth=data_cfg.get("min_depth", 1e-3),
        max_depth=data_cfg.get("max_depth", 10.0),
        min_depth_eval=data_cfg.get("min_depth_eval", 1e-3),
        max_depth_eval=data_cfg.get("max_depth_eval", 10.0),
    )
    eval_ds = NYUDepthDataset(
        data_path=data_cfg.eval_data_path,
        gt_path=data_cfg.eval_gt_path,
        filenames_file=data_cfg.get("filenames_file_eval", data_cfg.filenames_file),
        input_size=input_size,
        mode="val",
        use_dummy_data=data_cfg.use_dummy_data,
        n_samples=data_cfg.n_samples,
        do_random_rotate=False,
        degree=data_cfg.get("degree", 2.5),
        eigen_crop=data_cfg.get("eigen_crop", True),
        train_crop=data_cfg.get("train_crop", None),
        min_depth=data_cfg.get("min_depth", 1e-3),
        max_depth=data_cfg.get("max_depth", 10.0),
        min_depth_eval=data_cfg.get("min_depth_eval", 1e-3),
        max_depth_eval=data_cfg.get("max_depth_eval", 10.0),
    )

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed
        else None
    )
    eval_sampler = (
        DistributedSampler(eval_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if eval_distributed
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=eval_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, eval_loader, train_sampler, eval_sampler
