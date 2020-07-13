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
from modules import EfficientModel

def run():
    seed_torch(seed=config.seed)
    os.makedirs(config.MODEL_PATH,exist_ok=True)
    setup_logger(config.MODEL_PATH+'log.txt')
    writer = SummaryWriter(config.MODEL_PATH)

    folds = pd.read_csv(config.fold_csv)
    folds.head()
    if config.tile_stats_csv:
        attention_df = pd.read_csv(config.tile_stats_csv)
        attention_df.head()
    
    #train val split
    if config.DEBUG:
        folds = folds.sample(n=50, random_state=config.seed).reset_index(drop=True).copy()

    logging.info(f"fold: {config.fold}")
    fold = config.fold
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index
    df_train = folds.loc[trn_idx]
        
    df_val = folds.loc[val_idx]
    # #------single image------
    # train_dataset = PANDADataset(image_folder=config.DATA_PATH,
    #                             df=df_train,
    #                             image_size=config.IMG_SIZE,
    #                             num_tiles=config.num_tiles,
    #                             rand=False,
    #                             transform=get_transforms(phase='train'))
    # valid_dataset = PANDADataset(image_folder=config.DATA_PATH,
    #                             df=df_val,
    #                             image_size=config.IMG_SIZE,
    #                             num_tiles=config.num_tiles,
    #                             rand=False, 
    #                             transform=get_transforms(phase='valid'))

    #------image tiles------
    train_dataset = PANDADatasetTiles(image_folder=config.DATA_PATH,
                                df=df_train,
                                image_size=config.IMG_SIZE,
                                num_tiles=config.num_tiles,
                                transform=get_transforms(phase='train'),
                                attention_df= attention_df)
    valid_dataset = PANDADatasetTiles(image_folder=config.DATA_PATH,
                                df=df_val,
                                image_size=config.IMG_SIZE,
                                num_tiles=config.num_tiles,
                                transform=get_transforms(phase='valid'),
                                attention_df=attention_df)

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
    model=EnetNetVLAD(num_clusters=config.num_cluster,num_tiles=config.num_tiles,num_classes=config.num_class,arch='efficientnet-b4')
    #model = EfficientModel(c_out=6,n_tiles=config.num_tiles,
    #                       tile_size=config.IMG_SIZE,
    #                       name='efficientnet-b0',
    #                       strategy='bag',
    #                       head='attention')
    model = model.to(device)
    if config.multi_gpu:
        model = torch.nn.DataParallel(model)
    if config.ckpt_path:
        model.load_state_dict(torch.load(config.ckpt_path))
    warmup_factor = 10 
    warmup_epo = 1
    optimizer = Adam(model.parameters(), lr=config.lr/warmup_factor)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epoch-warmup_epo)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=warmup_factor, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)
    
    if config.apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)


    best_score = 0.
    best_loss = 100.
    if config.model_type=='reg':
        optimized_rounder = OptimizedRounder()
    optimizer.zero_grad()
    optimizer.step()
    for epoch in range(1,config.num_epoch+1):
        if scheduler:
            scheduler.step(epoch-1)
        if config.model_type!='reg':
            train_fn(train_loader,model,optimizer,device,epoch,writer,df_train)
            metric = eval_fn(val_loader,model,device,epoch,writer,df_val)
        else:
            coefficients =train_fn(train_loader,model,optimizer,device,epoch,writer,df_train,optimized_rounder)
            metric = eval_fn(val_loader,model,device,epoch,writer,df_val,coefficients)
        score = metric['score']
        val_loss = metric['loss']
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
