# Interpreting Pretrained Language Models via Concept Bottlenecks [PAKDD'24 (Oral)]

## Install

We follow installation instructions from the [CEBaB](https://github.com/CEBaBing/CEBaB.git) repository, which mainly depends on [Huggingface](https://github.com/huggingface/transformers.git).

## Experiments

The code is tested on NVIDIA 3090 and A100 40/80GB GPU. An example for running the experiments is as follows:

```shell
cd run_cebab
python cbm_joint.py
```

Note: It seems the random seed cannot control the randomness in parameter initialization in transformer, we suggest to run the code multiple times to get good scores.

## Citation
```
@article{tan2023interpreting,
  title={Interpreting Pretrained Language Models via Concept Bottlenecks},
  author={Tan, Zhen and Cheng, Lu and Wang, Song and Bo, Yuan and Li, Jundong and Liu, Huan},
  journal={arXiv preprint arXiv:2311.05014},
  year={2023}
}
```
