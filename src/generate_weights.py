import os 
import argparse
import skimage
import cv2 
import numpy as np 
import torch 
import pandas as pd 
from tqdm import tqdm
from torch.utils.data import Dataset
import config
from modules import EfficientModel
from dataset import PANDADatasetTiles,get_transforms


class  PANDADatasetTiles(Dataset):
    def __init__(self,image_folder,df,image_size,num_tiles,transform=None):
        self.image_folder = image_folder
        self.df = df.reset_index(drop=True)
        self.image_size = image_size 
        self.num_tiles = num_tiles
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row.image_id
        if config.tile_png:
            img_tiles = []
            for i in range(config.num_tiles):
                tile_path = os.path.join(self.image_folder,img_id,str(i)+'.png')
                tile = skimage.io.imread(tile_path)
                img_tiles.append(tile)
        else:
            if config.tiff:
                tiff_file = os.path.join(self.image_folder, f'{img_id}.tiff')
                image = skimage.io.MultiImage(tiff_file)[1]
            else:
                img_file = os.path.join(self.image_folder,f'{img_id}.png')
                image = skimage.io.imread(img_file)
                
            if config.crop_white:
                image = crop_white(image)

            if config.BRS:  
                img_tiles = get_tiles_brs(image,self.image_size,self.num_tiles)
            else:
                img_tiles = get_tiles(image,self.image_size,self.num_tiles)

        images = np.zeros((self.num_tiles,3,self.image_size,self.image_size),np.float32)
        for i,tile in enumerate(img_tiles):
            if self.transform:
                tile = self.transform(image=tile)['image']
            tile = tile.astype(np.float32)
            tile /=255. 
            tile = tile.transpose(2,0,1)
            images[i,:,:,:] = tile 

        return {'image':torch.tensor(images).float(),'name':img_id}


def load_model_from_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path)
    c_out = 1 
    model = EfficientModel(c_out, 64,256,'efficientnet-b0','bag','attention')
    model.load_state_dict(ckpt)
    model.eval()
    model.to('cuda')
    return model


def name_to_tile(name,num_tiles):
    tile_names = []
    for i in range(num_tiles):
        tile_name = name+'_'+str(i)+'.png'
        tile_names.append(tile_name)
    return tile_names


def compute_weights(model, dataset):
    names = []
    attention_weights = []
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            el = dataset[idx]
            image = el['image'].to('cuda')
            features = model.feature_extractor(image)
            features = features.view(-1, 64, features.shape[-1])
            weights = model.head.attention_pool.compute_weights(features)
            attention_weights.append(weights)
            names.append(el['name'])
    attention_weights = [weights.reshape(-1).cpu().detach().numpy() for weights in attention_weights]
    attention_weights = np.concatenate(attention_weights, axis=0)
    tile_names = [name_to_tile(name,64) for name in names]
    tile_names = np.concatenate(tile_names,axis=0) 
    img_names = [[name]*64 for name in names]
    img_names = np.concatenate(img_names,axis=0)
    return attention_weights,tile_names,img_names

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Compute an attention weight for every tile of every training slide.')
    parser.add_argument('--ckpt', required=True, help='attention-model checkpoint (.pth) written by train.py')
    parser.add_argument('--out', default=None, help='output csv (default: <config.tile_stats_csv without .csv>_new.csv, so the existing file is never overwritten)')
    args = parser.parse_args()
    out_csv = args.out or (os.path.splitext(config.tile_stats_csv)[0] + '_new.csv')

    folds = pd.read_csv(config.fold_csv)
    train_dataset = PANDADatasetTiles(image_folder=config.DATA_PATH,
                                df=folds,
                                image_size=256,
                                num_tiles=64,
                                transform=get_transforms(phase='valid'))
    
    #load model 
    model = load_model_from_ckpt(args.ckpt)
    weights,tile_names,img_names = compute_weights(model,train_dataset)
    tile_stats=pd.DataFrame({'image_id':img_names,'file_name':tile_names,'attention_fold_1':weights})
    tile_stats.to_csv(out_csv, index=False)

    

