
import torch
import torch.nn as nn
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as TF

from typing import Tuple

class RandomPatches(nn.Module):
    def __init__(self, size, npatch, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super(RandomPatches, self).__init__()
        
        self.crop = RandomCrop(size, padding, pad_if_needed, fill, padding_mode)
        
        self.npatch = npatch
        
    def forward(self, img):
        patches = []
        
        for i in range(self.npatch):
            patches.append(self.crop(img))
            
        patches = torch.stack(patches)
        
        return patches
    
class RandomPatchesIdentical(RandomCrop):
    def __init__(self, size, npatch, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)
        
        self.npatch = npatch
    
    @staticmethod
    def get_params(img, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        _, h, w = TF.get_dimensions(img)
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw
    
    def crop_img(self, img1, img2):
        if self.padding is not None:
            img1 = TF.pad(img1, self.padding, self.fill, self.padding_mode)
            img2 = TF.pad(img2, self.padding, self.fill, self.padding_mode)

        _, h1, w1 = TF.get_dimensions(img1)
        _, h2, w2 = TF.get_dimensions(img2)
        assert h1 == h2 and w1 == w2, f"Dimensions of input images should match. got {h1}x{w1} & {h2}x{w2}"
        # pad the width if needed
        if self.pad_if_needed and w1 < self.size[1]:
            padding = [self.size[1] - w1, 0]
            img1 = TF.pad(img1, padding, self.fill, self.padding_mode)
        if self.pad_if_needed and w2 < self.size[1]:
            padding = [self.size[1] - w1, 0]
            img2 = TF.pad(img2, padding, self.fill, self.padding_mode)
        
        # pad the height if needed
        if self.pad_if_needed and h1 < self.size[0]:
            padding = [0, self.size[0] - h1]
            img1 = TF.pad(img1, padding, self.fill, self.padding_mode)
        if self.pad_if_needed and h2 < self.size[0]:
            padding = [0, self.size[0] - h2]
            img2 = TF.pad(img2, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img1, self.size)
        img1 = TF.crop(img1, i, j, h, w)
        
        img2 = TF.crop(img2, i, j, h, w)

        return img1, img2
    
    def forward(self, img1, img2):
        patches1 = []
        patches2 = []
        
        for _ in range(self.npatch):
            p1, p2 = self.crop_img(img1, img2)
            patches1.append(p1)
            patches2.append(p2)
            
            
        patches1 = torch.stack(patches1)
        patches2 = torch.stack(patches2)
        
        return patches1, patches2