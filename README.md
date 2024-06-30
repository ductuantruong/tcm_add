# Temporal-Channel Modeling in Multi-head Self-Attention for Synthetic Speech Detection

This Repository contains the code and pretrained models for the following INTERSPEECH 2024 paper:

* **Title** : Temporal-Channel Modeling in Multi-head Self-Attention for Synthetic Speech Detection
* **Autor** : Duc-Tuan Truong, Ruijie Tao, Tuan Nguyen, Hieu-Thi Luong, Kong Aik Lee, Eng Siong Chng

## Pretrained Model
The pretrained model XLSR can be found at [link](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt).

We have uploaded pretrained models of our experiments. You can download pretrained models from [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/truongdu001_e_ntu_edu_sg/El7AV62BKkdKhOYCyB3s2EkBLr-aVdj0doH0HNj9mTIsGA?e=aOlRCB). 

## Training
The full training script will be uploaded soon after cleaning the code base.

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

[conformer](https://github.com/lucidrains/conformer) (for Conformer model architechture).

[DHVT](https://github.com/ArieSeirack/DHVT) (for Head Token desgin).

Thanks for these authors for sharing their work!