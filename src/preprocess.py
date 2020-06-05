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
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_save_path,image)    


def save_image_fix():
    orig_path = '/home/jijianyi/dataset/train_original_2/'
    save_path = '/home/jijianyi/dataset/train_original_2/'

    train_names = os.listdir(orig_path)
    for t in tqdm(train_names):
        img_name = t.split('.')[0]
        img_path = os.path.join(orig_path,t)
        img_save_path = os.path.join(save_path,img_name+'.png')
        image = cv2.imread(img_path) #actually is RGB
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
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
    df_train = pd.read_csv('./data/train.csv')
    df_train.head()
    df_train['fold'] = -1
    kf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed)
    for fold, (train_index, val_index) in enumerate(kf.split(df_train, df_train['isup_grade'])):
        df_train.loc[val_index, 'fold'] = int(fold)
    df_train.to_csv('./data/folds_v2.csv', index=None)
    df_train.head()


if __name__ == "__main__":
    train_val_split()
    # save_image()
    # save_crop_white()
    # save_image_fix()