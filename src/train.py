import os 
import logging
import pandas as pd 
import numpy as np 
from sklearn.model_selection import StratifiedKFold
import torch 
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
from torch.optim import Adam
from tensorboardX import SummaryWriter


import config 
if config.apex:
    from apex import amp
from utils import (
    setup_logger,save_dict_to_json,
    save_model,seed_torch,GradualWarmupScheduler)
from engine import train_fn,eval_fn,OptimizedRounder
from dataset import PANDADataset,PANDADatasetTiles,get_transforms
from model import * 

def run():
    seed_torch(seed=config.seed)
    os.makedirs(config.MODEL_PATH,exist_ok=True)
    setup_logger(config.MODEL_PATH+'log.txt')
    writer = SummaryWriter(config.MODEL_PATH)

    folds = pd.read_csv(config.fold_csv)
    folds.head()
    
    #train val split
    if config.DEBUG:
        folds = folds.sample(n=50, random_state=config.seed).reset_index(drop=True).copy()

    logging.info(f"fold: {config.fold}")
    fold = config.fold
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index
    df_train = folds.loc[trn_idx]
        
    df_val = folds.loc[val_idx]
    #------single image------
    train_dataset = PANDADataset(image_folder=config.DATA_PATH,
                                 df=df_train,
                                 image_size=config.IMG_SIZE,
                                 num_tiles=config.num_tiles,
                                 rand=False,
                                 transform=get_transforms(phase='train'))
    valid_dataset = PANDADataset(image_folder=config.DATA_PATH,
                                 df=df_val,
                                 image_size=config.IMG_SIZE,
                                 num_tiles=config.num_tiles,
                                 rand=False, 
                                 transform=get_transforms(phase='valid'))

    ##------image tiles------
    #train_dataset = PANDADatasetTiles(image_folder=config.DATA_PATH,
    #                             df=df_train,
    #                             image_size=config.IMG_SIZE,
    #                             num_tiles=config.num_tiles,
    #                             transform=get_transforms(phase='train'))
    #valid_dataset = PANDADatasetTiles(image_folder=config.DATA_PATH,
    #                             df=df_val,
    #                             image_size=config.IMG_SIZE,
    #                             num_tiles=config.num_tiles,
    #                             transform=get_transforms(phase='valid'))
    train_loader = DataLoader(train_dataset, 
                              batch_size=config.batch_size,
                              sampler=RandomSampler(train_dataset),
                              num_workers=8,
                              pin_memory=True)
    val_loader = DataLoader(valid_dataset, 
                            batch_size=config.batch_size,
                            sampler=SequentialSampler(valid_dataset),
                            num_workers=8,
                            pin_memory=True
                            )

    device = torch.device("cuda")
    model = SEResNeXt(num_classes=config.num_class)
    model = model.to(device)
    if config.multi_gpu:
        model = torch.nn.DataParallel(model)
    warmup_factor = 10 
    warmup_epo = 1
    optimizer = Adam(model.parameters(), lr=config.lr/warmup_factor)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epoch-warmup_epo)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=warmup_factor, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)

    
    if config.apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)


    best_score = 0.
    best_loss = 100.
    optimized_rounder = OptimizedRounder()
    optimizer.zero_grad()
    optimizer.step()
    for epoch in range(1,config.num_epoch+1):
        scheduler.step(epoch-1)
        coefficients =train_fn(train_loader,model,optimizer,device,epoch,writer,optimized_rounder,df_train)
        metric = eval_fn(val_loader,model,device,epoch,writer,df_val,coefficients)
        score = metric['score']
        val_loss = metric['loss']
        if score > best_score:
            best_score = score 
            logging.info(f"Epoch {epoch} - found best score {best_score} with coefficient {coefficients}")
            save_model(model,config.MODEL_PATH+f"best_kappa_f{fold}.pth")
        if val_loss < best_loss:
            best_loss = val_loss 
            logging.info(f"Epoch {epoch} - found best loss {best_loss}")
            save_model(model,config.MODEL_PATH+f"best_loss_f{fold}.pth")


if __name__=='__main__':
    run()
