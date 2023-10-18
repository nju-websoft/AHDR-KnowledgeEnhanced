#coding=utf-8
import json
import os
import torch
import numpy as np
import random

def save_dataset(path, dataset):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

def read_dataset(path):
    f = open(path, 'r', encoding='utf-8')
    dataset = json.load(f)
    if 'data' in dataset:
        dataset = dataset['data']
    return dataset

def save_model(output_model_file, model, optimizer):
    os.makedirs(output_model_file, exist_ok=True)
    output_model_file += 'pytorch_model.bin'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, output_model_file, _use_new_zipfile_serialization=False)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # gpu
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu
