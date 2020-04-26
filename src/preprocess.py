import pandas as pd 
import cv2 
import skimage.io 
import os
from tqdm import tqdm 

orig_path = '/mnt/data/prostate-cancer-grade-assessment/train_images/'
save_path = '/mnt/data/prostate-cancer-grade-assessment/train_256/'
os.mkdir(save_path)

train_names = os.listdir(orig_path)
for t in tqdm(train_names):
    img_name = t.split('.')[0]
    img_path = os.path.join(orig_path,t)
    img_save_path = os.path.join(save_path,img_name+'.png')
    image = skimage.io.MultiImage(img_path)
    image = cv2.resize(image[-1],(256,256)) 
    cv2.imwrite(img_save_path,image)    
