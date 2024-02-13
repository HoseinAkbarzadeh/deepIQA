from math import ceil
import os

import h5py as h5
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

from utils.transforms import RandomPatchesIdentical

class TID2013FR(Dataset):
    classsize = 120
    def __init__(self, dspath, rfpath, imgsize, npatch, classes=list(range(1,26)), 
                 transforms=[], mos_transforms=[], **kwargs) -> None:
        self.path = dspath
        self.rfpath = rfpath
        self.rngpatch = RandomPatchesIdentical(imgsize, npatch, **kwargs)
        
        self.classes = classes
        
        self.tr = Compose(transforms)
        self.mostr = Compose(mos_transforms)
        
    def __len__(self):
        return self.classsize*len(self.classes)
    
    def __getitem__(self, idx):
        cls = self.classes[idx//self.classsize]
        clsidx = idx%self.classsize
        
        ref = self.tr(Image.open(os.path.join(self.rfpath, f'I{cls:02d}.BMP')))
        
        with h5.File(self.path) as h5file:
            imgds = h5file[f'/distorted/i{cls:02d}_BMP']
            mosds = h5file[f'/distorted/i{cls:02d}_MOS']
            
            img = self.tr(imgds[clsidx].astype('u1'))
            mos = self.mostr(mosds[clsidx])
            
            img, ref = self.rngpatch(ToTensor()(img), ToTensor()(ref))
        
        return img, ref, mos