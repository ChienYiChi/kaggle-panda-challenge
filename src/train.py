import os 
import logging
import pandas as pd 
import numpy as np 
from sklearn.model_selection import StratifiedKFold
import torch 
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from apex import amp

import config 
from utils import (
    setup_logger,save_dict_to_json,
    save_model,seed_torch)
from engine import train_fn,eval_fn
from dataset import TrainDataset,get_transforms,TrainDatasetTiles
from efficientnet_pytorch import EfficientNet
from model import *


def run():
    seed_torch(seed=config.seed)
    os.makedirs(config.MODEL_PATH,exist_ok=True)
    setup_logger(config.MODEL_PATH+'log.txt')
    writer = SummaryWriter(config.MODEL_PATH)

    folds = pd.read_csv('./data/folds.csv')
    folds.head()
    
    #train val split
    if config.DEBUG:
        folds = folds.sample(n=20, random_state=config.seed).reset_index(drop=True).copy()

    logging.info(f"fold: {config.fold}")
    fold = config.fold
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index
        
    train_dataset = TrainDatasetTiles(config.DATA_PATH,
                                 folds.loc[trn_idx].reset_index(drop=True), 
                                 folds.loc[trn_idx].reset_index(drop=True)["isup_grade"], 
                                 transform=get_transforms(data='train'))
    valid_dataset = TrainDatasetTiles(config.DATA_PATH,
                                 folds.loc[val_idx].reset_index(drop=True), 
                                 folds.loc[val_idx].reset_index(drop=True)["isup_grade"], 
                                 transform=get_transforms(data='valid'))
    
    train_loader = DataLoader(train_dataset, 
                                batch_size=config.batch_size,
                                shuffle=True,
                                num_workers=8,
                                pin_memory=True)
    val_loader = DataLoader(valid_dataset, 
                            batch_size=config.batch_size,
                            num_workers=8,
                            pin_memory=True
                            )

    device = torch.device("cuda")
    model = Resnext50Tiles()    
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=config.lr, amsgrad=False)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True, eps=1e-6,min_lr=1e-7)
    
    if config.apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)


    best_score = 0.
    best_loss = 100.
    for epoch in range(config.num_epoch):
        train_fn(train_loader,model,optimizer,device,epoch,writer)
        metric = eval_fn(val_loader,model,device,epoch,writer)
        score = metric['score']
        val_loss = metric['loss']
        scheduler.step(val_loss)
        if score > best_score:
            best_score = score 
            logging.info(f"Epoch {epoch} - found best score {best_score}")
            save_model(model,config.MODEL_PATH+f"best_kappa_f{fold}.pth")
        if val_loss < best_loss:
            best_loss = val_loss 
            logging.info(f"Epoch {epoch} - found best loss {best_loss}")
            save_model(model,config.MODEL_PATH+f"best_loss_f{fold}.pth")


if __name__=='__main__':
    run()