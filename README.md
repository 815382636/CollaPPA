# CollaPPA

Collaborative Knowledge and Personalized Preference Alignment for Sequential Recommendation

## Requirements:

- Python 3.9.7
- PyTorch 1.10.1
- transformers 4.2.1
- tqdm
- numpy
- sentencepiece
- pyyaml

## Dataset

We apply the dataset preprocessed by [P5](https://github.com/jeykigung/P5) and [RDRec](https://github.com/WangXFng/RDRec). Download preprocessed data from this [Google Drive link](https://drive.google.com/file/d/1qGxgmx7G_WB7JE4Cn_bEcZ_o_NAJLE3G/view?usp=sharing), then put them into the _data_ folder.
[Attribute Generation Task](https://github.com/WangXFng/RDRec)

## Pretrained Checkpoints

Download pretrained Checkpoints from this [Google Drive link](https://drive.google.com/file/d/1qGxgmx7G_WB7JE4Cn_bEcZ_o_NAJLE3G/view?usp=sharing)

## Quick Start

### Multi-task Collaborative Fine-tuning

```
# example for Sports
sh sports.sh
```

### Personalized Preference Alignment

```
sh sports-PPA.sh
```

<!-- ## Citation

If the code and the paper are useful for you, it is appreciable to cite our paper:

```
@article{yue2025large,
  title={CoT4Rec: Unveiling User Preferences through Chain of Thought for Recommender Systems},
  author={Weiqi, Yue and Yuyu, Yin and Xin, Zhang and Binbin, Shi and Tingting, Liang and Jian, Wan},
  journal={proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
``` -->

## Thanks

The code refers to the repo [P5](https://github.com/jeykigung/P5), [POD](https://github.com/lileipisces/POD), [RDRec](https://github.com/WangXFng/RDRec) and [sdpo](https://github.com/chenyuxin1999/S-DPO).
