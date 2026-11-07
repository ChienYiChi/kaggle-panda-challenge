import os

# Where things live. Override with environment variables; the defaults work from a fresh checkout.
PANDA_ROOT = os.environ.get('PANDA_ROOT', './data/prostate-cancer-grade-assessment')  # competition download: train_images/*.tiff
WORK_DIR = os.environ.get('PANDA_WORK_DIR', './work')  # generated tiles, checkpoints and logs

DEBUG = False
model_type = "reg" # reg,cls,ord_reg
multi_gpu = True
tiff = False
tile_png = True
use_attention = True
apex = True
if multi_gpu:
    apex = False
IMG_SIZE = 256
num_tiles = 16
num_cluster = 6
fold = 1
backbone='regnet_800m'
pretrain_model = os.environ.get('PANDA_PRETRAIN_MODEL', os.path.join(WORK_DIR, 'ckpt', 'RegNetY-800MF_dds_8gpu.pyth'))
strategy='bag' #stitched,bag
fold_csv = os.environ.get('PANDA_FOLD_CSV', './data/train_clean.csv')
tile_stats_csv = os.environ.get('PANDA_TILE_STATS_CSV', './data/tile_stats.csv')
DATA_PATH = os.environ.get('PANDA_TILES_DIR', os.path.join(WORK_DIR, 'train_tiles'))
# MODEL_PATH is concatenated with file names elsewhere, so it must end with a path separator.
MODEL_PATH = os.path.join(WORK_DIR, 'logs', f'{strategy}-{backbone}-netvlad-attention-dropout-{model_type}-{num_tiles}-{IMG_SIZE}-f{fold}') + os.sep
num_folds = 5
seed=42
batch_size = 16
lr = 3e-4
num_epoch =30
num_class = 1
accumulation_steps = 32/batch_size
BRS = False
crop_white = False
ckpt_path = None
