import pandas as pd 
import numpy as np 
import cv2 
import skimage.io 
import os
import config
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm 

def crop_white(image: np.ndarray) -> np.ndarray:
    assert image.shape[2] == 3
    assert image.dtype == np.uint8
    ys, = (image.min((1, 2)) != 255).nonzero()
    xs, = (image.min(0).min(1) != 255).nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return image
    return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def save_image():
    orig_path = '/mnt/data/prostate-cancer-grade-assessment/train_images/'
    save_path = '/mnt/data/prostate-cancer-grade-assessment/train_original_2/'
    os.mkdir(save_path)

    train_names = os.listdir(orig_path)
    for t in tqdm(train_names):
        img_name = t.split('.')[0]
        img_path = os.path.join(orig_path,t)
        img_save_path = os.path.join(save_path,img_name+'.png')
        image = skimage.io.MultiImage(img_path)
        # image = cv2.resize(image[-1],(256,256)) 
        image = image[-2]
        cv2.imwrite(img_save_path,image)    


def save_crop_white():
    orig_path = '/mnt/data/prostate-cancer-grade-assessment/train_images/'
    save_path = '/mnt/data/prostate-cancer-grade-assessment/train_level_2_crop_white/'
    os.mkdir(save_path)

    train_names = os.listdir(orig_path)
    for t in tqdm(train_names):
        img_name = t.split('.')[0]
        img_path = os.path.join(orig_path,t)
        img_save_path = os.path.join(save_path,img_name+'.png')
        image = skimage.io.MultiImage(img_path)
        image = image[-2]
        image = crop_white(image)
        # image = cv2.resize(image,(512,512)) 
        cv2.imwrite(img_save_path,image)    


def train_val_split():
    train = pd.read_csv('./data/train.csv')
    train.head()
    folds = train.copy()

    train_labels = folds["isup_grade"].values
    kf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed)
    for fold, (train_index, val_index) in enumerate(kf.split(folds.values, train_labels)):
        folds.loc[val_index, 'fold'] = int(fold)
    folds['fold'] = folds['fold'].astype(int)
    folds.to_csv('./data/folds.csv', index=None)
    folds.head()


if __name__ == "__main__":
    # train_val_split()
    # save_image()
    save_crop_white()