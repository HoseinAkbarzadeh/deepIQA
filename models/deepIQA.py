from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepIQAOutput(OrderedDict):
    loss = None
    pooled_logits = None
    patches_logits = None
    def __init__(self, loss, pooled_logits, patches_logits):
        super(DeepIQAOutput, self).__init__()
        self.loss = loss
        self.pooled_logits = pooled_logits
        self.patches_logits = patches_logits
        
    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return tuple(self[k] for k in self.keys())
        
    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)
        
    def __repr__(self) -> str:
        return f"DeepIQAOuput({', '.join(f'{k}={v}' for k, v in self.__dict__.items())})"
        

class DeepIQA(nn.Module, ABC):
    def __init__(self):
        super(DeepIQA, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 'same')
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 'same')
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 'same')
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 'same')
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(64, 128, 3, 1, 'same')
        self.conv6 = nn.Conv2d(128, 128, 3, 1, 'same')
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv7 = nn.Conv2d(128, 256, 3, 1, 'same')
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 'same')
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.conv9 = nn.Conv2d(256, 512, 3, 1, 'same')
        self.conv10 = nn.Conv2d(512, 512, 3, 1, 'same')
        self.pool5 = nn.MaxPool2d(2, 2)
        
    def feature_extraction(self, x):
        # input shape: (batch_size, c, 32, 32)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool4(x)
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.pool5(x)
        return x.view(-1, 512)
    
    @abstractmethod
    def regression_and_pooling(self, patches, y_gt=None):
        pass
    
    def forward(self, gt, p, y_gt=None):
        # gt shape: (batch_size, num_patches, c, 32, 32)
        # p shape: (batch_size, num_patches, c, 32, 32)
        N, P, C, H, W = gt.size()
        assert p.size() == gt.size(), "gt and p should have the same shape"
        assert H == W == 32, "deepIQA only supports 32x32 patches"
        assert gt.dim() == 5, "gt and p should have 5 dimensions with (N, P, C, H, W) shape"
        
        gt = self.feature_extraction(gt.view(-1, C, H, W)).view(N, P, 512)
        p = self.feature_extraction(p.view(-1, C, H, W)).view(N, P, 512)
        
        loss, pooled, patches = self.regression_and_pooling(torch.cat([gt-p, gt, p], dim=-1), y_gt)
        
        return DeepIQAOutput(loss, pooled, patches)
    
class DIQaMFR(DeepIQA):
    def __init__(self, dropout=0.5) -> None:
        super(DIQaMFR, self).__init__()
        
        self.fc1 = nn.Linear(512*3, 512)
        self.fc2 = nn.Linear(512, 1)
        self.dp = nn.Dropout(dropout)
        
    def regression_and_pooling(self, patches, y_gt=None):
        N, P, _ = patches.size()
        patches = self.fc2(self.dp(F.relu(self.fc1(patches)))) # [N, P, 1]
        
        if y_gt is not None:
            assert y_gt.dim() == 1, "y_gt should have 1 dimensions as (N,)"
            
            y_gt = torch.repeat_interleave(y_gt[None,:,None], P, dim=1).reshape(N, P, 1)
            loss = torch.abs(y_gt - patches)
            loss = torch.mean(loss, dim=1, keepdim=True).squeeze(-1)
        else:
            loss = None
            
        pooled = torch.mean(patches, dim=1, keepdim=True).squeeze(-1)
        
        return loss, pooled, patches
    
class WaDIQaMFR(DeepIQA):
    epsilon = 1e-6
    def __init__(self, dropout=0.5) -> None:
        super(WaDIQaMFR, self).__init__()
        
        self.fc1 = nn.Linear(512*3, 512)
        self.fc2 = nn.Linear(512, 1)
        self.dp0 = nn.Dropout(dropout)
        
        self.wfc1 = nn.Linear(512*3, 512)
        self.wfc2 = nn.Linear(512, 1)
        self.dp1 = nn.Dropout(dropout)
        
    def regression_and_pooling(self, patches, y_gt=None):
        N, P, _ = patches.size()
        h = self.fc2(self.dp0(F.relu(self.fc1(patches)))) # [N, P, 1]
        alpha = F.relu(self.wfc2(self.dp1(F.relu(self.wfc1(patches))))) + self.epsilon
        alpha = alpha / torch.sum(alpha, dim=1, keepdim=True)
        
        patches = h*alpha # [N, P, 1]
        
        if y_gt is not None:
            assert y_gt.dim() == 1, "y_gt should have 1 dimensions as (N,)"
            loss = torch.abs(y_gt - torch.mean(patches, dim=1, keepdim=True).squeeze())
        else:
            loss = None
            
        pooled = torch.mean(patches, dim=1, keepdim=True).squeeze(-1)
        
        return loss, pooled, patches
            
            
class DIQaMNR(DeepIQA):
    def __init__(self, dropout=0.5):
        super(DIQaMNR, self).__init__()
        
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 1)
        self.dp = nn.Dropout(dropout)
        
    def forward(self, p, y_gt=None):
        # p shape: (batch_size, num_patches, c, 32, 32)
        N, P, C, H, W = p.size()
        assert H == W == 32, "deepIQA only supports 32x32 patches"
        assert p.dim() == 5, "gt and p should have 5 dimensions with (N, P, C, H, W) shape"
        
        p = self.feature_extraction(p.view(-1, C, H, W)).view(N, P, 512)
        
        loss, pooled, patches = self.regression_and_pooling(p, y_gt)
        
        return DeepIQAOutput(loss, pooled, patches)
    
    def regression_and_pooling(self, patches, y_gt=None):
        N, P, _ = patches.size()
        patches = self.fc2(self.dp(F.relu(self.fc1(patches)))) # [N, P, 1]
        
        if y_gt is not None:
            assert y_gt.dim() == 1, "y_gt should have 1 dimensions as (N,)"
            
            y_gt = torch.repeat_interleave(y_gt[None,:,None], P, dim=1).reshape(N, P, 1)
            loss = torch.abs(y_gt - patches)
            loss = torch.mean(loss, dim=1, keepdim=True).squeeze(-1)
        else:
            loss = None
            
        pooled = torch.mean(patches, dim=1, keepdim=True).squeeze(-1)
        
        return loss, pooled, patches
    
class WaDIQaMNR(DeepIQA):
    epsilon = 1e-6
    def __init__(self, dropout=0.5):
        super(WaDIQaMNR, self).__init__()
        
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 1)
        self.dp0 = nn.Dropout(dropout)
        
        self.wfc1 = nn.Linear(512, 512)
        self.wfc2 = nn.Linear(512, 1)
        self.dp1 = nn.Dropout(dropout)
        
    def forward(self, p, y_gt=None):
        # p shape: (batch_size, num_patches, c, 32, 32)
        N, P, C, H, W = p.size()
        assert H == W == 32, "deepIQA only supports 32x32 patches"
        assert p.dim() == 5, "gt and p should have 5 dimensions with (N, P, C, H, W) shape"
        
        p = self.feature_extraction(p.view(-1, C, H, W)).view(N, P, 512)
        
        loss, pooled, patches = self.regression_and_pooling(p, y_gt)
        
        return DeepIQAOutput(loss, pooled, patches)
    
    def regression_and_pooling(self, patches, y_gt=None):
        N, P, _ = patches.size()
        h = self.fc2(self.dp0(F.relu(self.fc1(patches)))) # [N, P, 1]
        alpha = F.relu(self.wfc2(self.dp1(F.relu(self.wfc1(patches))))) + self.epsilon
        alpha = alpha / torch.sum(alpha, dim=1, keepdim=True)
        
        patches = h*alpha # [N, P, 1]
        
        if y_gt is not None:
            assert y_gt.dim() == 1, "y_gt should have 1 dimensions as (N,)"
            loss = torch.abs(y_gt - torch.mean(patches, dim=1, keepdim=True).squeeze())
        else:
            loss = None
            
        pooled = torch.mean(patches, dim=1, keepdim=True).squeeze(-1)
        
        return loss, pooled, patches
    
        