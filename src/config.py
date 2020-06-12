DEBUG = False
multi_gpu = False
apex = True
if multi_gpu:
    apex = False
IMG_SIZE = 128
num_tiles = 36
fold = 0
fold_csv = './data/folds_v2.csv'
DATA_PATH = f'/mnt/data/prostate-cancer-grade-assessment/train_images/'
MODEL_PATH = f'/home/jijianyi/workspace/kaggle/logs/panda-challenge/seresnext50-reg-brs-{num_tiles}-{IMG_SIZE}-f{fold}/'

num_folds = 5
seed=42
batch_size = 16
lr = 3e-4
num_epoch =30
num_class = 1
