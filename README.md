# Temporal-Channel Modeling in Multi-head Self-Attention for Synthetic Speech Detection

This Repository contains the code and pretrained models for the following INTERSPEECH 2024 paper:

* **Title** : Temporal-Channel Modeling in Multi-head Self-Attention for Synthetic Speech Detection
* **Autor** : Duc-Tuan Truong, Ruijie Tao, Tuan Nguyen, Hieu-Thi Luong, Kong Aik Lee, Eng Siong Chng

## Pretrained Model
The pretrained model XLSR can be found at [link](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt).

We have uploaded pretrained models of our experiments. You can download pretrained models from [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/truongdu001_e_ntu_edu_sg/El7AV62BKkdKhOYCyB3s2EkBLr-aVdj0doH0HNj9mTIsGA?e=aOlRCB). 

## Setting up environment
Python version: 3.7.16

Install PyTorch
```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

Install other libraries:
```bash
pip install -r requirements.txt
```

Install fairseq:
```bash
git clone https://github.com/facebookresearch/fairseq.git fairseq_dir
cd fairseq_dir
git checkout a54021305d6b3c
pip install --editable ./
```

## Training & Testing on fixed-length input
To train and produce the score for LA set evaluation, run:
```bash
python main.py --algo 5
```

To train and produce the score for DF set evaluation, run:
```bash
python main.py --algo 3
```

## Inference
To run inference on a single wav file with the pretrained model, run:
```bash
python inference.py --ckpt_path=path_to/model.pth --threshold=-3.73 --wav_path=path_to/audio.flac
```
The threshold can be obtained when calculating EER on LA or DF set. In this example, the threshold is from DF set
evaluation.

## Citation
If you find our repository valuable for your work, please consider giving a start to this repo and citing our paper:
```
@misc{tcm,
      title={Temporal-Channel Modeling in Multi-head Self-Attention for Synthetic Speech Detection}, 
      author={Duc-Tuan Truong and Ruijie Tao and Tuan Nguyen and Hieu-Thi Luong and Kong Aik Lee and Eng Siong Chng},
      year={2024},
      eprint={2406.17376},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2406.17376}, 
}
```

### Acknowledge

Our work is built upon the [conformer-based-classifier-for-anti-spoofing](https://github.com/ErosRos/conformer-based-classifier-for-anti-spoofing) We also follow some parts of the following codebases:

[SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing) (for training pipeline).

[conformer](https://github.com/lucidrains/conformer) (for Conformer model architechture).

[DHVT](https://github.com/ArieSeirack/DHVT) (for Head Token desgin).

Thanks for these authors for sharing their work!
