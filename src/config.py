DEBUG = False
multi_gpu = False
apex = True
if multi_gpu:
    apex = False
IMG_SIZE = 256
num_tiles = 20
fold = 0
fold_csv = './data/folds.csv'
DATA_PATH = f'/mnt/data/prostate-cancer-grade-assessment/train_images/'
MODEL_PATH = f'/home/jijianyi/workspace/kaggle/logs/panda-challenge/tiles-eb0-cls-{num_tiles}-{IMG_SIZE}-f{fold}/'

num_folds = 5
seed=101
batch_size = 16
lr = 1e-4
num_epoch = 40
num_class = 6