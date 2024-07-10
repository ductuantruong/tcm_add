import torch
import argparse
from model import Model
import librosa

def load_wav_file(wav_path):
    wav, _ = librosa.load(wav_path, sr=16000)
    wav = torch.tensor(wav)
    return wav
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conformer-W2V')
    parser.add_argument('--threshold', type=float, default=-3.73, 
                    help='threshold score')
    parser.add_argument('--emb-size', type=int, default=144, metavar='N',
                    help='embedding size')
    parser.add_argument('--heads', type=int, default=4, metavar='N',
                    help='heads of the conformer encoder')
    parser.add_argument('--kernel_size', type=int, default=31, metavar='N',
                    help='kernel size conv module')
    parser.add_argument('--num_encoders', type=int, default=4, metavar='N',
                    help='number of encoders of the conformer')
    parser.add_argument('--wav_path', type=str, 
                    help='path to the wav file')
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

    # Loading input
    print('Model loaded : {}'.format(args.wav_path))
    wav = load_wav_file(args.wav_path).to(device)
    if len(wav.shape) == 1:
        wav = wav.unsqueeze(0)
    
    # Running inference
    with torch.no_grad():
        out, _ = model(wav)
    score = out[:, 1].item()
    print('Is the wav file bonafide? -> {}'.format(score > args.threshold))
