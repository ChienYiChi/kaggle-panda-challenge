DEBUG = False
multi_gpu = True
apex = True
if multi_gpu:
    apex = False
IMG_SIZE = 256
num_tiles =36
fold = 0
fold_csv = './data/folds_v2.csv'
DATA_PATH = f'/home/jijianyi/dataset/train_original_2/'
MODEL_PATH = f'/home/jijianyi/workspace/kaggle/logs/panda-challenge/resnext50-reg-{num_tiles}-{IMG_SIZE}-f{fold}/'

num_folds = 5
seed=42
batch_size = 4
lr = 1e-4
num_epoch = 40
num_class = 1
