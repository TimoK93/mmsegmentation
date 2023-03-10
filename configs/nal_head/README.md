# NAL (Noise Adaption Layer)

[Training deep neural-networks using a noise adaptation layer](https://openreview.net/forum?id=H12GRgcxg)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/udibr/noisy_labels">Official Repo</a>

## Abstract

<!-- [ABSTRACT] -->

The availability of large datasets has enabled neural networks to achieve impressive
recognition results. However, the presence of inaccurate class labels is known to
deteriorate the performance of even the best classifiers in a broad range of classi-
fication problems. Noisy labels also tend to be more harmful than noisy attributes.
When the observed label is noisy, we can view the correct label as a latent ran-
dom variable and model the noise processes by a communication channel with
unknown parameters. Thus we can apply the EM algorithm to find the parameters
of both the network and the noise and estimate the correct label. In this study we
present a neural-network approach that optimizes the same likelihood function as
optimized by the EM algorithm. The noise is explicitly modeled by an additional
softmax layer that connects the correct labels to the noisy ones. This scheme is
then extended to the case where the noisy labels are dependent on the features in
addition to the correct labels. Experimental results demonstrate that this approach
outperforms previous methods.

<!-- [IMAGE] -->

<object data="https://github.com/udibr/noisy_labels/raw/master/iclr17-poster.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/udibr/noisy_labels/raw/master/iclr17-poster.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/udibr/noisy_labels/raw/master/iclr17-poster.pdf">Download PDF</a>.</p>
    </embed>
</object>

## Citation

```bibtex
@inproceedings{
    goldberger2017training,
    title={Training deep neural-networks using a noise adaptation layer},
    author={Jacob Goldberger and Ehud Ben-Reuven},
    booktitle={International Conference on Learning Representations},
    year={2017},
    url={https://openreview.net/forum?id=H12GRgcxg}
}
```

## Usage

The Noise Adaption Layer (NAL) can be applied to every segmentation head that predicts semantic logits.
To implement the NAL, a wrapper is wrapped around a segmentation head.
During training, the predicted logits are mapped via the noise adaption layer to model existing label noise
and during the inference, the clean distribution is used.

To wrap a segmentation head, the configuration file needs to be modified according to the following code:

```python
_base_ = 'some_config.py'

model = dict(
    decode_head=dict(
        type='NoiseAdaptionLayerHead',
        model_type='simple_model',
        noise_initialisation=0.2,
        decode_head={{_base_.model.decode_head}}),
)
```

The parameter **model_type** can be ***simple_model*** or ***complex_model*** and **noise_initialisation** is the
start value of the uniform distributed confusion matrix. For further details, see the references paper.

## Results and models

### ADE20K

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU | mIoU(ms+flip) | config | download |
| ------ | -------- | --------- | ------: | -------- | -------------- | ---: | ------------: | ------ | -------- |
|        |          |           |         |          |                |      |               |        |          |
