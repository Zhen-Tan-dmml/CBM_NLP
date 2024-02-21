# Interpreting Pretrained Language Models via Concept Bottlenecks [PAKDD'24 (Oral)]
Please kindly check our new follow-up work accepted to [**AAAI'24**](https://arxiv.org/pdf/2312.15033.pdf): [**SpaseCBM**](https://github.com/Zhen-Tan-dmml/SparseCBM.git).

## Abstract
Pretrained language models (PLMs) have made significant strides in various natural language processing tasks. However, the lack of interpretability due to their "black-box" nature poses challenges for responsible implementation. Although previous studies have attempted to improve interpretability by using, e.g., attention weights in self-attention layers, these weights often lack clarity, readability, and intuitiveness.

In this research, we propose a novel approach to interpreting PLMs by employing ***high-level, meaningful concepts*** that are easily understandable for humans. For example, we learn the concept of "Food" and investigate how it influences the prediction of a model's sentiment towards a restaurant review. 

We introduce C^3M, which combines human-annotated and machine-generated concepts to extract hidden neurons designed to encapsulate semantically meaningful and task-specific concepts. Through empirical evaluations on real-world datasets, we show that our approach offers valuable insights to interpret PLM behavior, helps diagnose model failures, and enhances model robustness amidst noisy concept labels.

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
