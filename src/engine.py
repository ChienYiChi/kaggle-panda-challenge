import logging
import numpy as np 
import torch 
import torch.nn as nn 
from tqdm import tqdm 
from sklearn.metrics import cohen_kappa_score
import config 
from utils import accuracy
if config.apex:
    from apex import amp

threshold = [0.5,1.5,2.5,3.5,4.5]

def quadratic_weighted_kappa(y_hat, y):
    return cohen_kappa_score(y_hat, y, weights='quadratic')

#------ordinal regression------
#loss_fn = nn.BCEWithLogitsLoss()

#------regression------
def loss_fn(outputs,targets):
    targets = targets.view(-1,1).float()
    # return SmoothL1Loss()(outputs,targets)
    return nn.MSELoss()(outputs,targets)


def train_fn(data_loader,model,optimizer,device,epoch,writer):
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
        if config.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        pbar.set_description("loss:%.5f"%loss.item())
        avg_loss += loss.item()
    
    avg_loss /= len(data_loader)
    logging.info(f"Epoch {epoch} | train loss {avg_loss}")
    writer.add_scalar('train_loss',avg_loss,epoch+1)
    

def eval_fn(data_loader,model,device,epoch,writer,df):
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
        #------classification------
        # outputs = torch.softmax(outputs,dim=1)
        # fin_preds.extend(outputs.cpu().detach().numpy().argmax(1))

        #------ordinal regression------
        # outputs = outputs.sigmoid().sum(1).round()
        # labels = labels.sum(1)

        #------regression------
        # do noting to the output logits 

        fin_preds.extend(outputs.cpu().detach().numpy())
        fin_targets.extend(labels.cpu().detach().numpy())
    
    avg_loss /= len(data_loader)

    fin_targets = np.array(fin_targets)
    fin_preds = np.array(fin_preds)

    #------regression------
    fin_preds = np.concatenate(fin_preds)
    fin_preds[fin_preds<threshold[0]]=0
    fin_preds[(fin_preds>=threshold[0])&(fin_preds<threshold[1])]=1
    fin_preds[(fin_preds>=threshold[1])&(fin_preds<threshold[2])]=2
    fin_preds[(fin_preds>=threshold[2])&(fin_preds<threshold[3])]=3
    fin_preds[(fin_preds>=threshold[3])&(fin_preds<threshold[4])]=4
    fin_preds[fin_preds>=threshold[4]]=5
    
    qwk = quadratic_weighted_kappa(fin_targets,fin_preds)
    qwk_k = quadratic_weighted_kappa(fin_targets[df['data_provider']=='karolinska'],
                                    fin_preds[df['data_provider']=='karolinska'])
    qwk_r = quadratic_weighted_kappa(fin_targets[df['data_provider']=='radboud'],
                                    fin_preds[df['data_provider']=='radboud'])
    acc_metric = accuracy(fin_preds,fin_targets,6)
    
    logging.info(f"Epoch {epoch} | val loss {avg_loss} | kappa all:{qwk}, karolinska:{qwk_k}, radboud:{qwk_r}")
    logging.info(f"Epoch {epoch} | val acc {acc_metric}")
    writer.add_scalar("val_loss",avg_loss,epoch+1)
    writer.add_scalar("val_kappa",qwk,epoch+1)
    for i in range(config.num_class):
        writer.add_scalar(f'val/acc_{i}',acc_metric[i],epoch+1)

    metric = {"loss":avg_loss,"score":qwk}
    return metric


# def eval_fn(data_loader,model,device,epoch,writer,loss_fn):
#     model.eval()
#     fin_targets = []
#     fin_preds = []
#     avg_loss = 0.
    
#     for idx,data in tqdm(enumerate(data_loader),total=len(data_loader)):
#         images,labels = data
#         images = images.to(device)
#         labels = labels.to(device)

#         with torch.no_grad():
#             outputs = model(images)
#         loss = loss_fn(outputs,labels)
#         avg_loss += loss.item()
#         if config.num_class==6:
#             outputs = torch.softmax(outputs,dim=1)
#             fin_preds.extend(outputs.cpu().detach().numpy().argmax(1))
#         else:
#             fin_preds.extend(outputs.cpu().detach().numpy())

#         fin_targets.extend(labels.cpu().detach().numpy())
    
#     avg_loss /= len(data_loader)

#     fin_targets = np.array(fin_targets)
#     fin_preds = np.array(fin_preds)
#     if config.num_class==1:
#         fin_preds = np.concatenate(fin_preds)
#         fin_preds[fin_preds<threshold[0]]=0
#         fin_preds[(fin_preds>threshold[0])&(fin_preds<threshold[1])]=1
#         fin_preds[(fin_preds>threshold[1])&(fin_preds<threshold[2])]=2
#         fin_preds[(fin_preds>threshold[2])&(fin_preds<threshold[3])]=3
#         fin_preds[(fin_preds>threshold[3])&(fin_preds<threshold[4])]=4
#         fin_preds[fin_preds>threshold[4]]=5
#     score = quadratic_weighted_kappa(fin_targets,fin_preds)
#     acc_metric = accuracy(fin_preds,fin_targets,6)
    
#     logging.info(f"Epoch {epoch} | val loss {avg_loss} | kappa {score}")
#     logging.info(f"Epoch {epoch} | val acc {acc_metric}")
#     writer.add_scalar("val_loss",avg_loss,epoch+1)
#     writer.add_scalar("val_kappa",score,epoch+1)
#     for i in range(config.num_class):
#         writer.add_scalar(f'val/acc_{i}',acc_metric[i],epoch+1)

#     metric = {"loss":avg_loss,"score":score}
#     return metric
