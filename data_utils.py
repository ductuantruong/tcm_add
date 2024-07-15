import numpy as np
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from utils import pad
			

class Dataset_eval(Dataset):
    def __init__(self, list_IDs, base_dir, track):
        '''self.list_IDs	: list of strings (each string: utt key),'''
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 66800 # take ~4 sec audio 
        self.track = track
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, index):  
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir+'flac/'+utt_id+'.flac', sr=16000)
        X_pad = pad(X,self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id  
