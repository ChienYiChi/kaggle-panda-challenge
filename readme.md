# Kaggle Prostate cANcer graDe Assessment (PANDA) Challenge 

Here is the solution code of Team ChienYiChi for https://www.kaggle.com/c/prostate-cancer-grade-assessment/overview.

- kaggle profile: https://www.kaggle.com/ericji
- kaggle solution discussion: https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169114 

## Dependencies
`pip install -r requirements.txt`


## Experiments Record 

### tile images model
| type |  model  | private kappa |public kappa | local kappa  | karolinska kappa | radboud kappa |  fold num | image size | num tiles | epoch | TTA |
|:--------:|:--------:| :--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| cls | tiles-resnext50-netvlad | 0.896 | 0.879 | 0.8602 | 0.8884 | 0.8089 | 0 | 256 | 20 | 27 | 8 | 
| cls | tiles-eb0-netvlad | 0.882 | 0.849 | 0.8762 | 0.877 | 0.851 | 0 | 256 | 20 | 26 | 8 | 
| cls | tiles-eb0-netvlad | 0.857 | 0.856 | 0.8834 | 0.8714 | 0.8692 | 0 | 256 | 36 | 22 | 8 | 
| cls | tiles-resnet34-netvlad | 0.87 | 0.848 | 0.8745 | 0.8697 | 0.8522 | 0 | 256 | 20 | 28 | 8 | 
| reg | tiles-eb0-netvlad | 0.899 | 0.859 |0.8777 | 0.8820 | 0.8470 | 0 | 256 | 20 | 29 | 8 | 
| reg | tiles-eb0-netvlad | **0.920** | 0.881 | 0.8952 | 0.8976 | 0.8704 | 0 | 256 | 36 | 28 | 8 | 
| reg | tiles-eb0-netvlad | 0.903 | 0.893 | 0.886 | 0.8979 | 0.8464 | 1 | 256 | 36 | 22 | 8 | 
| reg | tiles-eb4-netvlad train with BRS（blue ratio selection）,test without BRS | 0.909 | **0.90** | 0.8826 | 0.9047 | 0.8335 | 1 | 256 | 36 | 26 | 8 | 
| reg | tiles-eb4-netvlad test with BRS | 0.913 | 0.896 | 0.8826 | 0.9047 | 0.8335 | 1 | 256 | 36 | 26 | 8 |
| reg | tiles-eb0-netvlad with attention model(128 tiles) to select tiles | 0.899 | 0.88 | 0.8833 | 0.9034 | 0.8367 | 1 | 256 | 16 | 27 | 8 TTA only for score model|
| reg | tiles-eb0-netvlad with attention model to select tiles | 0.910 | 0.88 | 0.8833 | 0.9034 | 0.8367 | 1 | 256 | 16 | 27 | 8 TTA only for score model|
| reg | tiles-eb0-netvlad with attention model to select tiles | 0.813 | 0.874 | 0.8859 | 0.8945 | 0.8481 | 1 | 256 | 36 | 25 | 8 TTA only for score model|
| reg | tiles-eb4-netvlad with attention model to select tiles | 0.908 | 0.897 | 0.8766 | 0.9035 | 0.8246 | 1 | 256 | 16 | 27 | 8 TTA only for score model|
| reg | newcv tiles-eb4-netvlad with attention model to select tiles | 0.904 | 0.899 | 0.8812 | 0.8958 | 0.8437 | 1 | 256 | 16 | 27 | 8 TTA only for score model|
| ord reg | newcv tiles-eb0-netvlad with attention model to select tiles | 0.913 | 0.884 | 0.8958 | 0.8972 | 0.8732 | 1 | 256 | 16 | 27 | 8 TTA only for score model|
| ord reg | newcv tiles-eb4-netvlad with attention model to select tiles | 0.900 | 0.879 | 0.887 | 0.9006 | 0.8524 | 1 | 256 | 16 | 27 | 8 TTA only for score model|
| reg | stitch-tiles-regnety_800m with attention model to select tiles | 0.911 | 0.893 | 0.8935 | 0.8872 | 0.8757 | 1 | 256 | 16 | 28 | 8 TTA only for score model |

## How to Run 
 - generate tiles using **preprocess.py**
### train one-stage model 
1. set model type and hyperparameters in **config.py**  
2. change model function in **train.py** 

### train two-stage model (attention model + score model)
1. set model type and hyperparameters in **config.py**
2. change the model function to the efficienet model with attention layer in **train.py**
3. generate tiles weights using **generate_weights.py** which will output a tiles weights csv file
4. set model type and hyperparameter again in **config.py** if you want to change the model type , for example regression or ordinal regression
5. change the model function in **train.py** ,for example, efficientnet with NetVlad layer


## References
Thanks everyone who shared their ideas on Kaggle discussion, I learned a lot from them.
- https://www.kaggle.com/iafoss/panda-concat-tile-pooling-starter-0-79-lb
- https://www.kaggle.com/haqishen/panda-inference-w-36-tiles-256
- https://github.com/facebookresearch/pycls
- https://github.com/loadder/netVLAD-pytorch 
- https://arxiv.org/abs/1511.07247
 
