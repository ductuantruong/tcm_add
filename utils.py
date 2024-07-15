import numpy as np
import os
import sys
import torch
import importlib
import random

def pad(x, max_len):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	

def read_metadata(dir_meta, is_eval=False):
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()
    
    if (is_eval):
        for line in l_meta:
            key= line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
             _,key,_,_,label = line.strip().split()
             
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list
    
def reproducibility(random_seed, args=None):                                  
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    cudnn_deterministic = True
    cudnn_benchmark = False
    print("cudnn_deterministic set to False")
    print("cudnn_benchmark set to True")
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
    return

def my_collate(batch): #Dataset return sample = (utterance, target, nameFile) #shape of utterance [1, lenAudio]
  data = [dp[0] for dp in batch]
  label = [dp[1] for dp in batch]
  nameFile = [dp[2] for dp in batch]
  return (data, label, nameFile) 
