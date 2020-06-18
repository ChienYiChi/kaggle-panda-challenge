import os 
import cv2
import numpy as np 
import skimage.io
import torch 
from torch.utils.data import Dataset 
import config 
import albumentations
from tqdm import tqdm 
import time 


class PANDADataset(Dataset):
    def __init__(self,
                 image_folder,   
                 df,
                 image_size,
                 num_tiles,
                 rand=False,
                 transform=None,
                ):
        self.image_folder = image_folder
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.num_tiles = num_tiles
        self.rand = rand
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row.image_id
        tiff_file = os.path.join(self.image_folder, f'{img_id}.tiff')
        image = skimage.io.MultiImage(tiff_file)[1]
        #img_file = os.path.join(self.image_folder,f'{img_id}.png')
        #image = cv2.imread(img_file)
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        tiles = get_tiles_brs(image, self.image_size,self.num_tiles)

        if self.rand:
            idxes = np.random.choice(list(range(self.num_tiles)), self.num_tiles, replace=False)
        else:
            idxes = list(range(self.num_tiles))

        n_row_tiles = int(np.sqrt(self.num_tiles))
        images = np.zeros((self.image_size * n_row_tiles, self.image_size * n_row_tiles, 3))
        for h in range(n_row_tiles):
            for w in range(n_row_tiles):
                i = h * n_row_tiles + w
    
                if len(tiles) > idxes[i]:
                    this_img = tiles[idxes[i]]
                else:
                    this_img = np.ones((self.image_size, self.image_size, 3)).astype(np.uint8) * 255
                this_img = 255 - this_img
                if self.transform is not None:
                    this_img = self.transform(image=this_img)['image']
                h1 = h * self.image_size
                w1 = w * self.image_size
                images[h1:h1+self.image_size, w1:w1+self.image_size] = this_img

        if self.transform is not None:
            images = self.transform(image=images)['image']
        images = images.astype(np.float32)
        images /= 255
        images = images.transpose(2, 0, 1)

        #------oridinal regression------
        #label = np.zeros(5).astype(np.float32)
        #label[:row.isup_grade] = 1.

        #------regression------
        label = row.isup_grade
        return torch.tensor(images).float(), torch.tensor(label).float()


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
        tiff_file = os.path.join(self.image_folder, f'{img_id}.tiff')
        image = skimage.io.MultiImage(tiff_file)[1]
        #img_file = os.path.join(self.image_folder,f'{img_id}.png')
        #image = cv2.imread(img_file)
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        img_tiles = get_tiles(image,self.image_size,self.num_tiles)
        images = np.zeros((self.num_tiles,3,self.image_size,self.image_size),np.float32)
        for i,tile in enumerate(img_tiles):
            if self.transform:
                tile = self.transform(image=tile)['image']
            tile = tile.astype(np.float32)
            tile /=255. 
            tile = tile.transpose(2,0,1)
            images[i,:,:,:] = tile 

        label = row.isup_grade
        
        return torch.tensor(images).float(), torch.tensor(label).float()


def blue_ratio_selection(img):
    hue = (100.*img[:,:,2])/(1.+img[:,:,0]+img[:,:,1])
    intensity = 256./(1.+img[:,:,0]+img[:,:,1]+img[:,:,2])
    blue_ratio = hue*intensity
    return blue_ratio


def get_tiles(img,sz,num_tiles):
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=255)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < num_tiles:
        img = np.pad(img,[[0,num_tiles-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:num_tiles]
    img = img[idxs]
    return img


def get_tiles_brs(img,sz,num_tiles):
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=255)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < num_tiles:
        img = np.pad(img,[[0,num_tiles-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:num_tiles*4]
    img = img[idxs]
    idxs = np.argsort([blue_ratio_selection(x).sum() for x in img])[::-1][:num_tiles]
    img = img[idxs]
    return img


def get_transforms(phase):
    if phase=='train':
        transform = albumentations.Compose([
            albumentations.Transpose(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
        ])
    else:
        transform = None
    return transform


if __name__=='__main__':
    pass

