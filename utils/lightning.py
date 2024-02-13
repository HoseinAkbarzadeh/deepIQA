

import torch
from lightning.pytorch import LightningModule

from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef

import models.deepIQA as deepIQA

class deepIQAFRLightning(LightningModule):
    def __init__(self, batchsize, lr, modelname='DIQaMFR', dropout=0.5):
        self.net = getattr(deepIQA, modelname)(dropout)
        
        self.save_hyperparameters()
        
    def forward(self, img1, img2, mos=None):
        return self.net(img1, img2, mos)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.hparams['lr'])
    
    def training_step(self, batch, batch_idx):
        disimg, refimg, mos = batch
        result = self(disimg, refimg, mos)
        loss = torch.mean(result.loss.squeeze())
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        disimg, refimg, mos = batch
        result = self(disimg, refimg, mos)
        loss = torch.mean(result.loss.squeeze())
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss
    
    def on_test_start(self):
        self.pred = []
        self.target = []
        
    def test_step(self, batch, batch_idx):
        disimg, refimg, mos = batch
        result = self(disimg, refimg, None)
        self.pred.append(result.pooled_logits.squeeze())
        self.target.append(mos.squeeze())
        
    def on_test_end(self):
        if self.global_rank == 0:
            self.pred = torch.cat(self.pred, dim=0)
            self.target = torch.cat(self.target, dim=0)
            
            rho = PearsonCorrCoef()
            spe = SpearmanCorrCoef()
            
            self.logger.experiment.add_hparams(
                {
                    'batch size': self.hparams['batchsize'],
                    'learning rate': self.hparams['lr']
                },
                {
                    'PLCC': rho(self.pred, self.target),
                    'SRCC': spe(self.pred, self.target)
                }
            )
        
        