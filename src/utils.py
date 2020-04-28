import os 
import logging
import sys
import json 
import numpy as np 
import random
import torch 
from torch.nn import DataParallel

LOGGER = logging.getLogger()
FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")


def setup_logger(out_file=None, stderr=True, stderr_level=logging.INFO, file_level=logging.DEBUG):
    LOGGER.handlers = []
    LOGGER.setLevel(min(stderr_level, file_level))

    if stderr:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(FORMATTER)
        handler.setLevel(stderr_level)
        LOGGER.addHandler(handler)

    if out_file is not None:
        handler = logging.FileHandler(out_file)
        handler.setFormatter(FORMATTER)
        handler.setLevel(file_level)
        LOGGER.addHandler(handler)

    LOGGER.info("logger set up")
    return LOGGER


def save_model(model, path):
    if isinstance(model, DataParallel):
        model = model.module

    with open(path, "wb") as fout:
        torch.save(model.state_dict(), fout)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def accuracy(preds,labels,num_classes):
    classes = range(num_classes)
    metric = {}
    for c in classes:
        pred = np.equal(c,preds)
        label = np.equal(c,labels)
        hit = pred&label
        pos = np.sum(label.astype(int))
        hit = np.sum(hit.astype(int))
        if pos==0:
            acc = 0.
        else:
            acc = hit/pos
        metric[c] = acc
    return metric

        
if __name__=='__main__':
    #unit test 
    preds = np.array([1,2,3,4,5,0,1,2])
    labels = np.array([1,2,2,1,1,1,1,1])
    print(accuracy(preds,labels,6))


