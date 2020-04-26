import os 
import logging
import pandas as pd 
import numpy as np 
from sklearn.model_selection import StratifiedKFold
import torch 
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import config 
from utils import (
    setup_logger,save_dict_to_json,
    save_model,seed_torch)
from engine import train_fn,eval_fn
from dataset import TrainDataset,get_transforms
from efficientnet_pytorch import EfficientNet



def run():
    seed_torch(seed=config.seed)
    os.makedirs(config.MODEL_PATH,exist_ok=True)
    setup_logger(config.MODEL_PATH+'log.txt')

    train = pd.read_csv('./data/train.csv')
    train.head()
    
    #train val split
    if config.DEBUG:
        folds = train.sample(n=20, random_state=config.seed).reset_index(drop=True).copy()
    else:
        folds = train.copy()

    train_labels = folds["isup_grade"].values
    kf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed)
    for fold, (train_index, val_index) in enumerate(kf.split(folds.values, train_labels)):
        folds.loc[val_index, 'fold'] = int(fold)
    folds['fold'] = folds['fold'].astype(int)
    folds.to_csv('folds.csv', index=None)
    folds.head()

    logging.info(f"fold: {config.fold}")
    fold = config.fold
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index
        
    train_dataset = TrainDataset(config.DATA_PATH,
                                 folds.loc[trn_idx].reset_index(drop=True), 
                                 folds.loc[trn_idx].reset_index(drop=True)["isup_grade"], 
                                 transform=get_transforms(data='train'))
    valid_dataset = TrainDataset(config.DATA_PATH,
                                 folds.loc[val_idx].reset_index(drop=True), 
                                 folds.loc[val_idx].reset_index(drop=True)["isup_grade"], 
                                 transform=get_transforms(data='valid'))
    
    train_loader = DataLoader(train_dataset, 
                                batch_size=config.batch_size,
                                shuffle=True,
                                num_workers=4,
                                pin_memory=True)
    val_loader = DataLoader(valid_dataset, 
                            batch_size=config.batch_size,
                            num_workers=4,
                            pin_memory=True
                            )

    device = torch.device("cuda")
    model = EfficientNet.from_pretrained("efficientnet-b4",num_classes=config.num_class)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=config.lr, amsgrad=False)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True, eps=1e-6,min_lr=1e-7)

    best_score = 0.
    for epoch in range(config.num_epoch):
        train_fn(train_loader,model,optimizer,device,epoch,scheduler)
        metric = eval_fn(val_loader,model,device,epoch)
        score = metric['score']
        if score > best_score:
            best_score = score 
            logging.info(f"Epoch {epoch} - found best score {best_score}")
            save_model(model,config.MODEL_PATH+f"best_fold{fold}.pth")


if __name__=='__main__':
    run()

