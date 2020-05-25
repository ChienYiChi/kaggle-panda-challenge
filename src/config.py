DEBUG = False
USE_TILES = True
apex = True
IMG_SIZE = 256
num_tiles = 20
fold = 1

if USE_TILES: 
    DATA_PATH = f'/mnt/data/prostate-cancer-grade-assessment/train_original_2/'
    MODEL_PATH = f'/home/jijianyi/workspace/kaggle/logs/panda-challenge/tiles-resnext50-cls-gem-{num_tiles}-{IMG_SIZE}-f{fold}/'
else:
    DATA_PATH = f'/mnt/data/prostate-cancer-grade-assessment/train_{IMG_SIZE}/'
    MODEL_PATH = f'/home/jijianyi/workspace/kaggle/logs/panda-challenge/eb04-reg-gem-{IMG_SIZE}-f{fold}/'
num_folds = 5
seed=101
batch_size = 8
lr = 1e-4
num_epoch = 40
num_class = 6
