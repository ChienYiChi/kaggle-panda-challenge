from tqdm import tqdm as tqdm
import os
import skimage.io
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision 
from efficientnet_pytorch import model as enet
import config
from sklearn.metrics import cohen_kappa_score
from functools import partial

ATTENTION_MODEL = '/home/jijianyi/workspace/kaggle/logs/panda-challenge/tiles-eb0-attention-cls-64-256-f1/best_kappa_f1.pth'
SCORE_MODEL = '/home/jijianyi/workspace/kaggle/logs/panda-challenge/tiles-eb0-netvlad-attention-reg-16-256-f1/best_kappa_f1.pth'
folds = pd.read_csv(config.fold_csv)
fold = config.fold
trn_idx = folds[folds['fold'] != fold].index
val_idx = folds[folds['fold'] == fold].index
df = folds.loc[val_idx]

image_folder = '/mnt/data/prostate-cancer-grade-assessment/train_images/'

tile_size = 256
batch_size = 4
num_tiles = 16
num_attention_tiles = 64
nworkers = 4

import torch.nn.init as init
import math
from torch.autograd import Variable


class NetVLAD(nn.Module):
    def __init__(self, feature_size, max_frames,cluster_size, add_bn=False, truncate=True):
        super(NetVLAD, self).__init__()
        self.feature_size = feature_size / 2 if truncate else feature_size
        self.max_frames = max_frames
        self.cluster_size = cluster_size
        self.batch_norm = nn.BatchNorm1d(cluster_size, eps=1e-3, momentum=0.01)
        self.linear = nn.Linear(self.feature_size, self.cluster_size)
        self.softmax = nn.Softmax(dim=1)
        self.cluster_weights2 = nn.Parameter(torch.FloatTensor(1, self.feature_size,
                                                               self.cluster_size))
        self.add_bn = add_bn
        self.truncate = truncate
        self.first = True
        self.init_parameters()

    def init_parameters(self):
        init.normal(self.cluster_weights2, std=1 / math.sqrt(self.feature_size))

    def forward(self, reshaped_input):
        random_idx = torch.bernoulli(torch.Tensor([0.5]))
        if self.truncate:
            if self.training == True:
                reshaped_input = reshaped_input[:, :self.feature_size].contiguous() if random_idx[0]==0 else reshaped_input[:, self.feature_size:].contiguous()
            else:
                if self.first == True:
                    reshaped_input = reshaped_input[:, :self.feature_size].contiguous()
                else:
                    reshaped_input = reshaped_input[:, self.feature_size:].contiguous()
        activation = self.linear(reshaped_input)
        if self.add_bn:
            activation = self.batch_norm(activation)
        activation = self.softmax(activation).view([-1, self.max_frames, self.cluster_size])
        a_sum = activation.sum(-2).unsqueeze(1)
        a = torch.mul(a_sum, self.cluster_weights2)
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = reshaped_input.view([-1, self.max_frames, self.feature_size])
        vlad = torch.matmul(activation, reshaped_input).permute(0, 2, 1).contiguous()
        vlad = vlad.sub(a).view([-1, self.cluster_size * self.feature_size])
        if self.training == False:
            self.first = 1 - self.first
        return vlad
    

class Model(nn.Module):
    def __init__(self, num_clusters,num_tiles,num_classes=1,arch='efficientnet-b0'):
        super().__init__()
        self.base = enet.EfficientNet.from_name(arch)
        self.nc = self.base._fc.in_features
        self.tile = nn.Sequential(
            nn.BatchNorm2d(self.nc,eps=0.001,momentum=0.01),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
        )    
        self.netvlad = NetVLAD(cluster_size=num_clusters,max_frames=num_tiles,
                    feature_size=self.nc,truncate=False)
        self.fc = nn.Linear(num_clusters*self.nc,num_classes)
    
    def forward(self, x):
        """
        Args:
            x (batch,N,3,h,w):
        """
        batch = x.shape[0]
        shape = x[0].shape
        n = shape[0]
        x = x.reshape(-1,shape[1],shape[2],shape[3]) #x: bs*num_tiles x 3 x H x W
        x = self.base.extract_features(x) #x: bs*num_tiles x nc
        x = self.tile(x)
        x = x.view(batch,n,self.nc)
        x = self.netvlad(x)
        x = self.fc(x)
        return x


state_dict = torch.load(SCORE_MODEL,map_location=lambda storage, loc: storage)
model = Model(num_clusters=6,num_tiles=num_tiles,num_classes=1,arch='efficientnet-b0')
model.load_state_dict(state_dict,strict=True)
model.eval()
model.cuda()

del state_dict


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self, x):
        avg_x = self.avg(x)
        max_x = self.max(x)
        return torch.cat([avg_x, max_x], dim=1).squeeze(2).squeeze(2)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class EfficientModel(nn.Module):

    def __init__(self, c_out=6, n_tiles=36, tile_size=224, name='efficientnet-b0', strategy='stitched', head='basic'):
        super().__init__()

        from efficientnet_pytorch import EfficientNet
        m = EfficientNet.from_name(name,override_params={"num_classes":c_out})
        c_feature = m._fc.in_features
        m._fc = nn.Identity()
        self.feature_extractor = m
        self.n_tiles = n_tiles
        self.tile_size = tile_size
        
        if strategy == 'stitched':
            if head == 'basic':
                self.head = nn.Linear(c_feature, c_out)
            elif head == 'concat':
                m._avg_pooling = AdaptiveConcatPool2d()
                self.head = nn.Linear(c_feature * 2, c_out)
            elif head == 'gem':
                m._avg_pooling = GeM()
                self.head = nn.Linear(c_feature, c_out)
        elif strategy == 'bag':
            if head == 'basic':
                self.head = BasicHead(c_feature, c_out, n_tiles)
            elif head == 'attention':
                self.head = AttentionHead(c_feature, c_out, n_tiles)
                
        self.strategy = strategy

    def forward(self, x):
        if self.strategy == 'bag':
            x = x.view(-1, 3, self.tile_size, self.tile_size)
        h = self.feature_extractor(x)
        h = self.head(h)
        return h


class BasicHead(nn.Module):

    def __init__(self, c_in, c_out, n_tiles):
        self.n_tiles = n_tiles
        super().__init__()
        self.fc = nn.Sequential(AdaptiveConcatPool2d(),
                                nn.Linear(c_in * 2, c_out))

    def forward(self, x):

        bn, c = x.shape
        h = x.view(-1, self.n_tiles, c, 1, 1).permute(0, 2, 1, 3, 4) \
            .contiguous().view(-1, c, 1 * self.n_tiles, 1)
        h = self.fc(h)
        return h


class AttentionHead(nn.Module):

    def __init__(self, c_in, c_out, n_tiles):
        self.n_tiles = n_tiles
        super().__init__()
        self.attention_pool = AttentionPool(c_in, c_in//2)
        self.fc = nn.Linear(c_in, c_out)

    def forward(self, x):

        bn, c = x.shape
        h = x.view(-1, self.n_tiles, c)
        h = self.attention_pool(h)
        h = self.fc(h)
        return h


class AttentionPool(nn.Module):

    def __init__(self, c_in, d):
        super().__init__()
        self.lin_V = nn.Linear(c_in, d)
        self.lin_w = nn.Linear(d, 1)

    def compute_weights(self, x):
        key = self.lin_V(x)  # b, n, d
        weights = self.lin_w(torch.tanh(key))  # b, n, 1
        weights = torch.softmax(weights, dim=1)
        return weights

    def forward(self, x):
        weights = self.compute_weights(x)
        pooled = torch.matmul(x.transpose(1, 2), weights).squeeze(2)   # b, c, n x b, n, 1 => b, c, 1
        return pooled

state_dict = torch.load(ATTENTION_MODEL,map_location=lambda storage, loc: storage)
attention_model = EfficientModel(c_out=6,
                                 n_tiles=num_tiles, 
                                 tile_size=tile_size, 
                                 name='efficientnet-b0', 
                                 strategy='bag', 
                                 head='attention')
attention_model.load_state_dict(state_dict,strict=True)
attention_model.float()
attention_model.eval()
attention_model.cuda()

del state_dict


import scipy
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

optimized_rounder = OptimizedRounder()


def preprocess(img,sz,num_tiles):
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=255)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:num_tiles]
    img = img[idxs]
    return img



class PandaDataset(Dataset):
    def __init__(self,path,test_df,attention_model):
        self.path = path 
        self.df = test_df
        self.attention_model = attention_model
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        row = self.df.iloc[idx]
        name = row.image_id
        image = skimage.io.MultiImage(os.path.join(self.path,name+'.tiff'))[1]
        tiles = preprocess(image,tile_size,num_attention_tiles)
        images = np.zeros((num_attention_tiles,3,tile_size,tile_size),np.float32)
        for i,tile in enumerate(tiles):
            tile = tile.astype(np.float32)
            tile /=255. 
            tile = tile.transpose(2,0,1)
            images[i,:,:,:] = tile 
        best_tiles = self.get_best_tiles(images)

        label = row.isup_grade
        label = torch.tensor(label).float()
        return torch.tensor(best_tiles).float(),name,label
    
    def get_best_tiles(self,tiles):
        x = torch.tensor(tiles)
        x = torch.stack([x,x.flip(-1),x.flip(-2),x.flip(-1,-2),
            x.transpose(-1,-2),x.transpose(-1,-2).flip(-1),
            x.transpose(-1,-2).flip(-2),x.transpose(-1,-2).flip(-1,-2)],0)
        weights = 0
        for tile_bags in x:
            with torch.no_grad():
                features = self.attention_model.feature_extractor(tile_bags.cuda())
                features = features.view(-1, num_attention_tiles, features.shape[-1])
                weights += self.attention_model.head.attention_pool.compute_weights(features)
        weights /= len(x)
        weights = weights.reshape(-1).argsort(0).cpu().numpy()[::-1]
        best_tiles = [tiles[w] for w in weights[:num_tiles]] 
        return best_tiles

def quadratic_weighted_kappa(y_hat, y):
    return cohen_kappa_score(y_hat, y, weights='quadratic')

threshold = [0.5206,1.5448,2.4244,3.4923,4.3578]
ds = PandaDataset(image_folder,df,attention_model)
names,preds,labels = [],[],[]

with torch.no_grad():
    for el in tqdm(ds):
        x,y,label = el
        x = x.cuda()
        #dihedral TTA
        x = torch.stack([x,x.flip(-1),x.flip(-2),x.flip(-1,-2),
            x.transpose(-1,-2),x.transpose(-1,-2).flip(-1),
            x.transpose(-1,-2).flip(-2),x.transpose(-1,-2).flip(-1,-2)],0)
        # x = x.view(-1,num_tiles,3,tile_size,tile_size)
        p = model(x)
        p = p.view(-1).mean().cpu()
        names.append(y)
        preds.append(p)
        labels.append(label)

    fin_preds = np.array(preds)
    fin_targets = np.array(labels)
    optimized_rounder.fit(fin_preds,fin_targets)
    fin_preds = optimized_rounder.predict(fin_preds,threshold)
    qwk = quadratic_weighted_kappa(fin_targets,fin_preds)
    qwk_k = quadratic_weighted_kappa(fin_targets[df['data_provider']=='karolinska'],
                                    fin_preds[df['data_provider']=='karolinska'])
    qwk_r = quadratic_weighted_kappa(fin_targets[df['data_provider']=='radboud'],
                                    fin_preds[df['data_provider']=='radboud'])
    
    print(f"kappa all:{qwk}, karolinska:{qwk_k}, radboud:{qwk_r}")


