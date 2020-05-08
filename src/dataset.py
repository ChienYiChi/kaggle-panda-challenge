import cv2
import numpy as np 
import skimage.io
import torch 
from torch.utils.data import Dataset 
from albumentations import Compose, Normalize, HorizontalFlip, VerticalFlip,Rotate
from albumentations.pytorch import ToTensorV2
import config 


class TrainDataset(Dataset):
    def __init__(self,root_path,df, labels, transform=None):
        self.root_path = root_path
        self.df = df
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['image_id'].values[idx]
        # file_path = f'{self.root_path}{file_name}.tiff'
        # image = skimage.io.MultiImage(file_path)
        # image = cv2.resize(image[-1], (config.IMG_SIZE, config.IMG_SIZE))

        file_path = f'{self.root_path}{file_name}.png'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        label = self.labels[idx]
        
        return image, label


class TestDataset(Dataset):
    def __init__(self, root_path,df, dir_name, transform=None):
        self.root_path = root_path
        self.df = df
        self.dir_name = dir_name
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['image_id'].values[idx]
        file_path = f'{self.root_path}{file_name}.tiff'
        image = skimage.io.MultiImage(file_path)
        image = cv2.resize(image[-1], (config.IMG_SIZE, config.IMG_SIZE))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image


def get_transforms(*, data):
    
    assert data in ('train', 'valid')
    
    if data == 'train':
        return Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Rotate(limit=15,p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
    elif data == 'valid':
        return Compose([
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

def tile(img, mask):
    sz = config.IMG_SIZE
    N = config.num_tiles
    result = []
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=255)
    mask = np.pad(mask,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=0)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    mask = mask.reshape(mask.shape[0]//sz,sz,mask.shape[1]//sz,sz,3)
    mask = mask.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        mask = np.pad(mask,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=0)
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    mask = mask[idxs]
    for i in range(len(img)):
        result.append({'img':img[i], 'mask':mask[i], 'idx':i})
    return result


def tile_image(img):
    sz = config.IMG_SIZE
    N = config.num_tiles
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=255)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    return img


class TrainDatasetTiles(Dataset):
    def __init__(self,root_path,df, labels, transform=None):
        self.root_path = root_path
        self.df = df
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['image_id'].values[idx]
        # file_path = f'{self.root_path}{file_name}.tiff'
        # image = skimage.io.MultiImage(file_path)
        # image = cv2.resize(image[-1], (config.IMG_SIZE, config.IMG_SIZE))

        file_path = f'{self.root_path}{file_name}.png'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        img_tiles = tile_image(image)
        images = []
        if self.transform:
            for i in range(len(img_tiles)):
                augmented = self.transform(image=img_tiles[i])
                images.append(augmented['image'].unsqueeze(0))

        images = torch.cat(images,dim=0)
        label = self.labels[idx]
        
        return images, label