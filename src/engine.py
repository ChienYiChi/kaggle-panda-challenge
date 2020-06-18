import logging
import numpy as np 
import torch 
import torch.nn as nn 
from tqdm import tqdm 
from sklearn.metrics import cohen_kappa_score
import scipy
from functools import partial

import config 
from utils import accuracy
if config.apex:
    from apex import amp

threshold = [0.5,1.5,2.5,3.5,4.5]

def quadratic_weighted_kappa(y_hat, y):
    return cohen_kappa_score(y_hat, y, weights='quadratic')

#------classification------
# loss_fn = nn.CrossEntropyLoss()

#------ordinal regression------
# loss_fn = nn.BCEWithLogitsLoss()

#------regression------
def loss_fn(outputs,targets):
    outputs = outputs.view(-1)
    # return nn.SmoothL1Loss()(outputs,targets)
    return nn.MSELoss()(outputs,targets)

#------regression------
def train_fn(data_loader,model,optimizer,device,epoch,writer,optimized_rounder,df):
#------classification------
# def train_fn(data_loader,model,optimizer,device,epoch,writer,df):
    model.train()
    fin_preds = []
    fin_targets = []
    pbar = tqdm(enumerate(data_loader),total=len(data_loader))
    avg_loss = 0.
    for idx,data in pbar:
        images,labels = data
        images = images.to(device) 
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs,labels)
        if config.accumulation_steps!=1:
            loss = loss/config.accumulation_steps
        if config.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
        else:
            loss.backward()
        # Every iters_to_accumulate iterations, call step() and reset gradients:
        if (config.accumulation_steps!=1) and (idx%config.accumulation_steps == 0):
            optimizer.step()
            optimizer.zero_grad()
        optimizer.step()
        pbar.set_description("loss:%.5f"%loss.item())
        avg_loss += loss.item()
        # ------regression------
        fin_preds.append(outputs.cpu().detach().numpy())
        fin_targets.append(labels.cpu().detach().numpy())
    
    fin_preds = np.concatenate(fin_preds)
    fin_targets = np.concatenate(fin_targets)
    optimized_rounder.fit(fin_preds,fin_targets)
    coefficients = optimized_rounder.coefficients()
    fin_preds = optimized_rounder.predict(fin_preds,coefficients)
    qwk = quadratic_weighted_kappa(fin_targets,fin_preds)
    qwk_k = quadratic_weighted_kappa(fin_targets[df['data_provider']=='karolinska'],
                                    fin_preds[df['data_provider']=='karolinska'])
    qwk_r = quadratic_weighted_kappa(fin_targets[df['data_provider']=='radboud'],
                                    fin_preds[df['data_provider']=='radboud'])
    avg_loss /= len(data_loader)
    #------regression------
    logging.info(f"Epoch {epoch} | lr {optimizer.param_groups[0]['lr']:.7f} | train loss {avg_loss} | kappa all:{qwk}, karolinska:{qwk_k}, radboud:{qwk_r} | coefficients: {coefficients}")
    #------classification------
    # logging.info(f"Epoch {epoch} | lr {optimizer.param_groups[0]['lr']:.7f} | train loss {avg_loss} ")

    writer.add_scalar('train_loss',avg_loss,epoch+1)
    #------regression------
    return coefficients
    
##------regression------
def eval_fn(data_loader,model,device,epoch,writer,df,coefficients):
#------classification-------
# def eval_fn(data_loader,model,device,epoch,writer,df):
    optimized_rounder = OptimizedRounder()
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
        # #------classification------
        # outputs = torch.softmax(outputs,dim=1)
        # fin_preds.extend(outputs.cpu().detach().numpy().argmax(1))

        # #------ordinal regression------
        # outputs = outputs.sigmoid().sum(1).round()
        # labels = labels.sum(1)

        #------regression------
        # do noting to the output logits 

        fin_preds.extend(outputs.cpu().detach().numpy())
        fin_targets.extend(labels.cpu().detach().numpy())
    
    avg_loss /= len(data_loader)

    fin_targets = np.array(fin_targets)
    fin_preds = np.array(fin_preds)
    
    # #------regression------
    # fin_preds = np.concatenate(fin_preds)
    # fin_preds[fin_preds<threshold[0]]=0
    # fin_preds[(fin_preds>=threshold[0])&(fin_preds<threshold[1])]=1
    # fin_preds[(fin_preds>=threshold[1])&(fin_preds<threshold[2])]=2
    # fin_preds[(fin_preds>=threshold[2])&(fin_preds<threshold[3])]=3
    # fin_preds[(fin_preds>=threshold[3])&(fin_preds<threshold[4])]=4
    # fin_preds[fin_preds>=threshold[4]]=5

    fin_preds = optimized_rounder.predict(fin_preds,coefficients)
    
    qwk = quadratic_weighted_kappa(fin_targets,fin_preds)
    qwk_k = quadratic_weighted_kappa(fin_targets[df['data_provider']=='karolinska'],
                                    fin_preds[df['data_provider']=='karolinska'])
    qwk_r = quadratic_weighted_kappa(fin_targets[df['data_provider']=='radboud'],
                                    fin_preds[df['data_provider']=='radboud'])
    acc_metric = accuracy(fin_preds,fin_targets,config.num_class)
    
    logging.info(f"Epoch {epoch} | val loss {avg_loss} | kappa all:{qwk}, karolinska:{qwk_k}, radboud:{qwk_r}")
    logging.info(f"Epoch {epoch} | val acc {acc_metric}")
    writer.add_scalar("val_loss",avg_loss,epoch+1)
    writer.add_scalar("val_kappa",qwk,epoch+1)
    for i in range(config.num_class):
        writer.add_scalar(f'val/acc_{i}',acc_metric[i],epoch+1)

    metric = {"loss":avg_loss,"score":qwk}
    return metric


class OptimizedRounder():
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            else:
                X_p[i] = 5

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5]
        self.coef_ = scipy.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            else:
                X_p[i] = 5
        return X_p

    def coefficients(self):
        return self.coef_['x']