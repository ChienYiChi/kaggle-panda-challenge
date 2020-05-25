# Kaggle Prostate cANcer graDe Assessment (PANDA) Challenge 

Solution code for https://www.kaggle.com/c/prostate-cancer-grade-assessment/overview.

## Dependencies
`pip install -r requirements.txt`


## To-Do-List
- [x] find a stable validation set
- [x] White Background Trim Function
- [x] compare best loss ckpt and best kappa ckpt
- [x] find the best image size 
- [x] 5 fold ensemble 
- [x] find best model 

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
|  model  | local kappa | public kappa  |  fold num | image size | num tiles |
|:--------:| :--------:|:--------:|:--------:|:--------:|:--------:|
| resnext50 | 0.7910 | 0.75 | 0 | 128 | 12 |
| resnext50 | 0.8379 | 0.83 | 0 | 256 | 12 | 
| resnext50 | 0.8474 | 0.86 | 0 | 256 | 20 |
| resnext50 | 0.8635|  0.86 | 0 | 256 | 32 |
| resnetx50 | 0.8600 | 0.85 | 0 | 256 | 20 | 
| resnext50 | **0.8800** | 0.8497 | 1 | 256 | 20 | 
