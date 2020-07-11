# Kaggle Prostate cANcer graDe Assessment (PANDA) Challenge 

Solution code for https://www.kaggle.com/c/prostate-cancer-grade-assessment/overview.

## Dependencies
`pip install -r requirements.txt`


## Experiments Record 

### E1: Classification vs Regression 
| type     |  model  |  local kappa | public kappa  |  fold num | image size |
| :--------: |:--------:| :--------:|:--------:|:--------:|:--------:|
|classification| eb04 | 0.7406 | 0.59 | 0 | 512 |
|classification| eb04 | 0.7406 | 0.59 | 0 | 512 |
|classification| eb04 3rd epoch ckpt | 0.6848 | 0.59 | 0 | 512 | 
|classification| eb04 8th epoch ckpt | 0.7098 | 0.54 | 0 | 512 |
|regression| eb04 | 0.727 | 0.50 | 0 | 256 | 
|regression| eb04 | 0.7136| 0.53 | 1 | 256 |

### E2: White Background Trim Function
| type     |  model  |  local kappa | public kappa  |  fold num | image size |
| :--------: |:--------:| :--------:|:--------:|:--------:|:--------:|
|classification| eb04 | 0.7410 | 0.54 | 0 | 512 |

### E3: Image size
| type     |  model  |  local kappa | public kappa  |  fold num | image size |
| :--------: |:--------:| :--------:|:--------:|:--------:|:--------:|
|classification| eb04 |  |  | 0 | 768 |
|classification| eb04 |  |  | 0 | 1024 |

### E4: Image Tiles Input
|  model  | public kappa | local kappa  |  fold num | image size | num tiles | TTA | 
|:--------:| :--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| resnext50 | 0.75 | 0.7916 | 0 | 128 | 12 | 8 | 
| resnext50 | 0.83 | 0.8379 | 0 | 256 | 12 | 8 | 
| resnext50 | 0.86 | 0.8474 | 0 | 256 | 20 | 8 |
| resnext50 | 0.86|  0.8635 | 0 | 256 | 32 | 8 |
| resnetx50 | 0.85 | 0.8600 | 0 | 256 | 20 | 8 |
| resnext50 | 0.88 | 0.8497 | 1 | 256 | 20 | 8 |
| resnext50 | 0.85 |0.8497 | 1 | 256 | 20 | 0 | 
| eb0 | 0.83 | 0.8411 | 0 | 256 | 20 | 8 | 

| type |  model  | public kappa | local all kappa  | karolinska kappa | radboud kappa |  fold num | image size | num tiles | epoch | TTA |
|:--------:|:--------:| :--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| cls | tiles-resnext50-netvlad | 0.87 | 0.8602 | 0.8884 | 0.8089 | 0 | 256 | 20 | 27 | 8 | 
| cls | tiles-eb0-netvlad | 0.84 | 0.8762 | 0.877 | 0.851 | 0 | 256 | 20 | 26 | 8 | 
| cls | tiles-eb0-netvlad | 0.85 | 0.8834 | 0.8714 | 0.8692 | 0 | 256 | 36 | 22 | 8 | 
| cls | tiles-resnet34-netvlad | 0.84 | 0.8745 | 0.8697 | 0.8522 | 0 | 256 | 20 | 28 | 8 | 
| reg | tiles-eb0-netvlad | 0.85 | 0.8777 | 0.8820 | 0.8470 | 0 | 256 | 20 | 29 | 8 | 
| reg | tiles-eb0-netvlad | 0.88 | 0.8952 | 0.8976 | 0.8704 | 0 | 256 | 36 | 28 | 8 | 
| reg | tiles-eb0-netvlad | 0.89 | 0.886 | 0.8979 | 0.8464 | 1 | 256 | 36 | 22 | 8 | 
| reg | tiles-eb4-netvlad | **0.90** | 0.8826 | 0.9047 | 0.8335 | 1 | 256 | 36 | 26 | 8 | 
| cls | tiles-eb0-attention | | 0.8651 | 0.8867 | 0.8152 | 1 | 256 | 64 | 22 | |
| reg | tiles-eb0-netvlad with model above to select tiles | 0.88 | 0.8833 | 0.9034 | 0.8367 | 1 | 256 | 16 | 27 | 8 TTA only for score model|  


### E6: 36 x tiles 256
| type |  model  | public kappa | local all kappa  | karolinska kappa | radboud kappa |  fold num | image size | num tiles | epoch | TTA |
|:--------:|:--------:| :--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| ordinal regression| eb0 | 0.84 | 0.91 ||| 0 | 256 | 36 | | 0 |
| ordinal regression  | eb0 version2 | 0.84 | 0.8653 | 0.8758 | 0.822 | 0 | 256 | 36 | 22 | 0 | 
| ordinal regression | eb0 | 0.85 | 0.8645 | 0.8826,| 0.8114 | 1 | 256 | 36 | 30 | 0 |
| ordinal regression | eb0 | 0.85 | 0.8645 | 0.8826 | 0.8114 | 1 | 256 | 36 | 30 | 8 |
| ordinal regression | eb0 new ranadom seed 42 | 0.85 | 0.8751 | 0.8746 | 0.8435 | 0 | 256 | 36 | 30 | 0 | 
| regression | eb0  new ranadom seed 42 | 0.86 | 0.9029 | 0.8866 | 0.8784 | 0 | 256 | 36 | 28 | 0 | 
| regression | eb0  new random seed 42 | 0.87 | 0.9029 | 0.8866 | 0.8784 | 0 | 256 | 36 | 28 | 8 | 

 
