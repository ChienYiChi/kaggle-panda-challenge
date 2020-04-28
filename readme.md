# Kaggle Prostate cANcer graDe Assessment (PANDA) Challenge 

Solution code for https://www.kaggle.com/c/prostate-cancer-grade-assessment/overview.

## Dependencies
`pip install -r requirements.txt`


## To-Do-List
- [ ] find a stable validation set
- [x] White Background Trim Function
- [ ] compare best loss ckpt and best kappa ckpt
- [ ] find the best image size 
- [ ] 5 fold ensemble 
- [ ] find best model 

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