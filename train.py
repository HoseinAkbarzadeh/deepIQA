
import os

import torch
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from utils.lightning import deepIQAFRLightning
from data.tid2013 import TID2013FR

slurm = False

def parse_cmd():
    from argparse import ArgumentParser
    
    parser = ArgumentParser('deepIQA training and reporting the test results')
    
    parser.add_argument('--dspath', type=str, required=True, 
                        help='Path to TID2013 HDF5 file')
    parser.add_argument('--rfpath', type=str, required=True, 
                        help='Path to TID2013 reference images folder')
    parser.add_argument('--imgsize', type=int, default=32, 
                        help='Image size')
    parser.add_argument('--npatch', type=int, default=32,
                        help='Number of patches')
    parser.add_argument('--batchsize', type=int, default=3,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--modelname', type=str, default='DIQaMFR',
                        help='Model name')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Max epochs')
    parser.add_argument('--devices', type=int, default=1,
                        help='Number of GPUs')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for dataloaders')
    
    return parser.parse_args()

def main(args):
    
    seed_everything(42)
    
    classes = torch.arange(1, 26)
    rndidx = torch.randperm(len(classes), generator=torch.Generator().manual_seed(42))
    train_classes = rndidx[:15]
    val_classes = rndidx[15:20]
    test_classes = rndidx[20:]
    
    train_ds = TID2013FR(args.dspath, args.rfpath, args.imgsize, args.npatch, 
                         classes=train_classes, transforms=[], mos_transforms=[])
    val_ds = TID2013FR(args.dspath, args.rfpath, args.imgsize, args.npatch, 
                       classes=val_classes, transforms=[], mos_transforms=[])
    
    train_loader = DataLoader(train_ds, batch_size=args.batchsize, pin_memory=True,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batchsize, pin_memory=True,
                            shuffle=False, num_workers=args.num_workers)
    
    model = deepIQAFRLightning(args.batchsize, args.lr, args.modelname, args.dropout)
    
    proj_dir = os.environ['PROJECT_DIR'] if slurm else 'project_dir'
    
    
    ckpt = ModelCheckpoint(proj_dir, monitor='val_loss', mode='min', 
                           save_last=True, save_top_k=1)
    
    tblogger = TensorBoardLogger(proj_dir, name=args.modelname)
    
    trainer = Trainer(
        accelerator='gpu',
        strategy='ddp',
        devices=args.devices,
        callbacks=[RichProgressBar(), ckpt],
        logger=tblogger, 
        max_epochs=args.max_epochs
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    print("Finished training.")
    
    test_ds = TID2013FR(args.dspath, args.rfpath, args.imgsize, args.npatch, 
                       classes=test_classes, transforms=[], mos_transforms=[])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, 
                             pin_memory=True, num_workers=0)
    
    trainer.test(model, test_loader)
    
    print("Finished the test")
    
if __name__ == '__main__':
    args = parse_cmd()
    if 'SLURM_JOB_ID' in os.environ:
        slurm = True
    main(args)
    