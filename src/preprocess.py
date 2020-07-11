import pandas as pd 
import numpy as np 
import cv2 
import skimage.io 
import os
import config
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm 
import multiprocessing
import dataset
import glob 


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


def images_to_tiles(slide_paths,save_path,img_size,num_tiles,start_ind,end_ind):
    for sp in slide_paths[start_ind:end_ind+1]:
        name = sp.split('/')[-1].split('.')[0]
        img = skimage.io.MultiImage(sp)[1]
        shape = img.shape
        best_tiles = dataset.get_tiles(img,img_size,num_tiles)
        tile_root_path = os.path.join(save_path,name)
        os.mkdir(tile_root_path)
        for i in range(num_tiles):
            tile_path = os.path.join(tile_root_path,str(i)+'.png')
            skimage.io.imsave(tile_path,best_tiles[i])
    return (start_ind,end_ind) 


def multiprocess_make_images_tiles():
    TILE_SIZE = 256 
    NUM_TILES = 64
    orig_path = '/mnt/data/prostate-cancer-grade-assessment/train_images/'
    save_path = '/mnt/data/prostate-cancer-grade-assessment/train_tiles/'
    os.mkdir(save_path)

    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_processes)     
    
    train_image_pathes = glob.glob(orig_path+"*.tiff")
    num_train_images = len(train_image_pathes)
    images_per_process = num_train_images/num_processes

    tasks = []
    for num_process in range(1,num_processes+1):
        start_index = (num_process -1)*images_per_process+1
        end_index = num_process * images_per_process
        start_index = int(start_index)
        end_index = int(end_index)
        tasks.append((train_image_pathes,save_path,TILE_SIZE,NUM_TILES,start_index,end_index))
        if start_index == end_index:
            print("Task #" + str(num_process) + ": Process slide " + str(start_index))
        else:
            print("Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

    #start tasks
    results = []
    for t in tasks:
        results.append(pool.apply_async(images_to_tiles, t))
    
    for result in results:
        (start_ind,end_ind) = result.get()
        if start_ind==end_ind:
            print("Done converting slide %d" % start_ind)
        else:
            print("Done converting slides %d through %d" % (start_ind,end_ind))


if __name__ == "__main__":
    multiprocess_make_images_tiles()
