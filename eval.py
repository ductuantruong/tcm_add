import argparse
import sys
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from data_utils import Dataset_eval
from model import Model
from utils import reproducibility
from utils import read_metadata
import numpy as np
from tqdm import tqdm


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)
    model.eval()
    fname_list = []
    score_list = []
    text_list = []

    for batch_x,utt_id in data_loader:
        batch_x = batch_x.to(device)
        batch_out, _ = model(batch_x)
        batch_score = (batch_out[:, 1]
                       ).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    for f, cm in zip(fname_list, score_list):
        text_list.append('{} {}'.format(f, cm))
    del fname_list
    del score_list
    with open(save_path, 'a+') as fh:
        for i in range(0, len(text_list), 500):
            batch = text_list[i:i+500]
            fh.write('\n'.join(batch) + '\n')
    del text_list
    fh.close()
    print('Scores saved to {}'.format(save_path))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conformer-W2V')
    parser.add_argument('--database_path', type=str, default='ASVspoof_database/', help='Change this to user\'s directory address of LA database')
    parser.add_argument('--protocols_path', type=str, default='ASVspoof_database/', help='Change with path to user\'s LA database protocols directory address')
    parser.add_argument('--emb-size', type=int, default=144, metavar='N',
                    help='embedding size')
    parser.add_argument('--heads', type=int, default=4, metavar='N',
                    help='heads of the conformer encoder')
    parser.add_argument('--kernel_size', type=int, default=31, metavar='N',
                    help='kernel size conv module')
    parser.add_argument('--num_encoders', type=int, default=4, metavar='N',
                    help='number of encoders of the conformer')
    parser.add_argument('--ckpt_path', type=str, 
                    help='path to the model weigth')
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))

    # Loading model
    model = Model(args,device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()
    print('Model loaded : {}'.format(args.ckpt_path))

    eval_tracks=['LA','DF']

    for tracks in eval_tracks:
        prefix      = 'ASVspoof_{}'.format(tracks)
        prefix_2019 = 'ASVspoof2019.{}'.format(tracks)
        prefix_2021 = 'ASVspoof2021.{}'.format(tracks)

        file_eval = read_metadata( dir_meta =  os.path.join(args.protocols_path+'{}/{}_cm_protocols/{}.cm.eval.trl.txt'.format(tracks, prefix,prefix_2021)), is_eval=True)
        print('no. of eval trials',len(file_eval))
        eval_set=Dataset_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path+'{}/ASVspoof2021_{}_eval/'.format(tracks,tracks)),track=tracks)
        produce_evaluation_file(eval_set, model, device, 'Scores/{}/{}.txt'.format(tracks, args.ckpt_path.replace('/', '_')))
