# Bayesian Deeplabv3+

This implementation is based on the paper
[Evaluating Bayesian Deep Learning Methods for Semantic Segmentation](https://arxiv.org/abs/1811.12709)
but does only implement the concepts of the bayesian deep neural network.
To get the exact implementation as described in the paper,
further implementations and parameterization needs to be done.

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/fregu856/evaluating_bdl">Official Repo</a>

## Abstract

<!-- [ABSTRACT] -->

Deep learning has been revolutionary for computer vision and semantic segmentation in particular, with Bayesian Deep Learning (BDL) used to obtain uncertainty maps from deep models when predicting semantic classes. This information is critical when using semantic segmentation for autonomous driving for example. Standard semantic segmentation systems have well-established evaluation metrics. However, with BDL's rising popularity in computer vision we require new metrics to evaluate whether a BDL method produces better uncertainty estimates than another method. In this work we propose three such metrics to evaluate BDL models designed specifically for the task of semantic segmentation. We modify DeepLab-v3+, one of the state-of-the-art deep neural networks, and create its Bayesian counterpart using MC dropout and Concrete dropout as inference techniques. We then compare and test these two inference techniques on the well-known Cityscapes dataset using our suggested metrics. Our results provide new benchmarks for researchers to compare and evaluate their improved uncertainty quantification in pursuit of safer semantic segmentation.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-1811-12709,
  author    = {Jishnu Mukhoti and
               Yarin Gal},
  title     = {Evaluating Bayesian Deep Learning Methods for Semantic Segmentation},
  journal   = {CoRR},
  volume    = {abs/1811.12709},
  year      = {2018},
  url       = {http://arxiv.org/abs/1811.12709},
  eprinttype = {arXiv},
  eprint    = {1811.12709},
  timestamp = {Mon, 03 Dec 2018 07:50:28 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1811-12709.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Usage

The Bayesian Neural Network (BNN) is implemented in the well known DeepLabv3+ Framework.
But in general, the BNN can be applied to every encoder-decoder segmentation framework
that uses a supported backbone.
To use a BNN, dropout needs to be applied during inference in the backbone of the model.

To modify a framework to a BNN, the configuration file needs to be modified according to the
following code:

```python
_base_ = 'some_config.py'

model = dict(
    type='BayesianEncoderDecoder',
    iterations=10,
    backbone=dict(dropout_rates=(None, None, 0.25, None)))
```

The parameter **iterations** denotes the number of inference steps done per sample to gather the
mean and variance.

## Results and models

### ADE20K

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU | mIoU(ms+flip) | config | download |
| ------ | -------- | --------- | ------: | -------- | -------------- | ---: | ------------: | ------ | -------- |
|        |          |           |         |          |                |      |               |        |          |
