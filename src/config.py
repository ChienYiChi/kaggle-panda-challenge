DEBUG = False
USE_TILES = True
multi_gpu = False
apex = True
if multi_gpu:
    apex = False
IMG_SIZE = 256
num_tiles = 36
fold = 0
fold_csv = './data/folds_v2.csv'
if USE_TILES: 
    DATA_PATH = f'/mnt/data/prostate-cancer-grade-assessment/train_images/'
    MODEL_PATH = f'/home/jijianyi/workspace/kaggle/logs/panda-challenge/eb0-reg-{num_tiles}-{IMG_SIZE}-f{fold}/'
else:
    DATA_PATH = f'/mnt/data/prostate-cancer-grade-assessment/train_{IMG_SIZE}/'
    MODEL_PATH = f'/home/jijianyi/workspace/kaggle/logs/panda-challenge/eb04-reg-{IMG_SIZE}-f{fold}/'
num_folds = 5
seed=42
batch_size = 8
lr = 1e-4
num_epoch = 40
num_class = 1