import logging
import numpy as np 
import torch 
from torch.nn import CrossEntropyLoss,SmoothL1Loss
from tqdm import tqdm 
from sklearn.metrics import cohen_kappa_score
import config 
from utils import accuracy

threshold = [0.5,1.5,2.5,3.5,4.5]

def quadratic_weighted_kappa(y_hat, y):
    return cohen_kappa_score(y_hat, y, weights='quadratic')


def loss_fn(outputs,targets):
    assert config.num_class==1 or config.num_class==6
    if config.num_class==6:
        return CrossEntropyLoss()(outputs,targets)
    else:
        targets = targets.view(-1,1).float()
        return SmoothL1Loss()(outputs,targets)
    

def train_fn(data_loader,model,optimizer,device,epoch,scheduler,writer):
    model.train()

    pbar = tqdm(enumerate(data_loader),total=len(data_loader))
    avg_loss = 0.
    for idx,data in pbar:
        images,labels = data
        images = images.to(device) 
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs,labels)
        loss.backward()
        optimizer.step()
        pbar.set_description("loss:%.5f"%loss.item())
        avg_loss += loss.item()
    
    avg_loss /= len(data_loader)
    if scheduler:
        scheduler.step(avg_loss)

    logging.info(f"Epoch {epoch} | train loss {avg_loss}")
    writer.add_scalar('train_loss',avg_loss,epoch+1)
    

def eval_fn(data_loader,model,device,epoch,writer):
    model.eval()
    fin_targets = []
    fin_preds = []
    avg_loss = 0.
    
    for idx,data in tqdm(enumerate(data_loader),total=len(data_loader)):
        images,labels = data
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)
        loss = loss_fn(outputs,labels)
        avg_loss += loss.item()
        if config.num_class==6:
            outputs = torch.softmax(outputs,dim=1)
            fin_preds.extend(outputs.cpu().detach().numpy().argmax(1))
        else:
            fin_preds.extend(outputs.cpu().detach().numpy())

        fin_targets.extend(labels.cpu().detach().numpy())
    
    avg_loss /= len(data_loader)

    fin_targets = np.array(fin_targets)
    fin_preds = np.array(fin_preds)
    if config.num_class==1:
        fin_preds[fin_preds<threshold[0]]=0
        fin_preds[(fin_preds>threshold[0])&(fin_preds<threshold[1])]=1
        fin_preds[(fin_preds>threshold[1])&(fin_preds<threshold[2])]=2
        fin_preds[(fin_preds>threshold[2])&(fin_preds<threshold[3])]=3
        fin_preds[(fin_preds>threshold[3])&(fin_preds<threshold[4])]=4
        fin_preds[fin_preds>threshold[4]]=5

    score = quadratic_weighted_kappa(fin_targets,fin_preds)
    acc_metric = accuracy(fin_preds,fin_targets,config.num_class)
    
    logging.info(f"Epoch {epoch} | val loss {avg_loss} | kappa {score}")
    logging.info(f"Epoch {epoch} | val acc {acc_metric}")
    writer.add_scalar("val_loss",avg_loss,epoch+1)
    writer.add_scalar("val_kappa",score,epoch+1)
    for i in range(config.num_class):
        writer.add_scalar(f'val/acc_{i}',acc_metric[i],epoch+1)

    metric = {"loss":avg_loss,"score":score}
    return metric
