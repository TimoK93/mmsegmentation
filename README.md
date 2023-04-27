# Compensation Learning

Compensation Learning in Semantic Segmentation:

![](network.png)

> [**Compensation Learning in Semantic Segmentation**](https://arxiv.org/abs/2304.13428),
> Timo Kaiser, Christoph Reinders, Bodo Rosenhahn
> *arXiv report ([arXiv 2304.13428](https://arxiv.org/abs/2304.13428))*

```
@inproceedings{kaiser_2023_CVPR,
    title={Compensation Learning in Semantic Segmentation},
    author = {Kaiser, Timo and Reinders, Christoph and Rosenhahn, Bodo},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    year={2023}
}
```

Contact: [kaiser@tnt.uni-hannover.de](kaiser@tnt.uni-hannover.de). Any questions or discussion are welcome!

## Abstract

Label noise and ambiguities between similar classes are
challenging problems in developing new models and annotating new data for semantic segmentation. In this paper,
we propose Compensation Learning in Semantic Segmentation, a framework to identify and compensate ambiguities
as well as label noise. More specifically, we add a ground
truth depending and globally learned bias to the classification logits and introduce a novel uncertainty branch for
neural networks to induce the compensation bias only to relevant regions. Our method is employed into state-of-the-art
segmentation frameworks and several experiments demonstrate that our proposed compensation learns inter-class relations
that allow global identification of challenging ambiguities as well as the exact localization of subsequent label noise.
Additionally, it enlarges robustness against label
noise during training and allows target-oriented manipulation during inference. We evaluate the proposed method on
Cityscapes, KITTI-STEP, ADE20k, and COCO-stuff10k.

## Installation

This repository is built as a fork on top of the [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) repository. Please refer to [get_started.md](docs/en/get_started.md) for
installation instructions.
After installation, you can use the scripts described below.

Data preparation is described in [dataset_prepare.md](docs/en/dataset_prepare.md). Please note that the dataset
KITTI-STEP is currently only supported by this repository and not by MMSegmentation. We are working on a pull request!

## Usage

We provide and explain the following features from the paper in this repository:

- Training with compensation
- Biased Inference with compensation
- Plot compensation values
- Plot certainty distribution
- Uncertainty estimation
- Competitors:
  - Bayesian Neural Network
  - Noise Adaption Layer: Simple and Complex noise modeling
  - LogComp
- Pretrained models

### Training with Compensation

All reported training schedules can be found in the folder **configs/compensation_head**. For example, to train a
compensation head on KITTI-STEP with a deeplabv3plus backbone, run

```shell
tools/dist_train.sh configs/compensation_head/compensation_head_deeplabv3plus_r50-d8_368x368_80k_kittistep.py 8
```

Compensation can be applied via the CompensationHead class which is a wrapper around the original decode head.
The CompensationHead class can be found in **mmseg/models/decode_heads/compensation_head.py**.
The config file for a training schedule is shown below:

```python
_base_ = '../deeplabv3plus/deeplabv3plus_r50-d8_512x512_80k_ade20k.py'

model = dict(
    decode_head=dict(
        type='CompensationHead',
        local_compensation=True,
        loss_balancing=0.01,
        non_diagonal=True,
        symmetric=True,
        top_k=5,
        decode_head={{_base_.model.decode_head}},
    ),
)
```

The following parameters can be set in the config file:

- non_diagonal: If true, the diagonal entries of the compensation matrix are forced to be zero.
- symmetric: If true, the compensation matrix is symmetric.
- top_k: The top k most frequent classes are used to compute the uncertainty (not important for training or inference!).
- loss_balancing: The loss balancing factor for the compensation loss.
- local_compensation: If true, the local compensation branch is activated in the decode head.

NOTE: The training schedules in this repository are defined for 4 GPUs. If you use a larger or smaller GPU setup, you have to adjust the batch size
in the configuration file.

We used the batch sizes of the original deeplabv3plus/segformer implementations. The batch size for the datasets are
shown in the following table:

| Method        | Dataset    | Batch size |
| ------------- | ---------- | ---------- |
| deeplabv3plus | Cityscapes | 16         |
| deeplabv3plus | KITTI-STEP | 16         |
| deeplabv3plus | ADE20k     | 32         |
| deeplabv3plus | COCO-stuff | 32         |
| segformer     | Cityscapes | 8          |
| segformer     | KITTI-STEP | 8          |
| segformer     | ADE20k     | 16         |
| segformer     | COCO-stuff | 16         |

NOTE: Evaluation, inference, and other "standard" tasks are performed with the original MMSegmentation codebase.
Please read the MMSegmentation documentation for detailed information. To simply evaluate a model, run

```shell
python tools/test.py configs/compensation_head/compensation_head_deeplabv3plus_r50-d8_368x368_80k_kittistep.py your_model.pth --eval mIoU
```

### Biased Inference with compensation

First, you need to train a compensation aware segmentation network. To induce bias into the inference,
have a look into the config file
**configs/compensation_head/induction_compensation_head_deeplabv3plus_r50-d8_368x368_80k_kittistep.py**.
The bias during inference is induced with the field **induction_weights** that is a list of tuples (i, j, value), shown
as follows:

```python
model = dict(
    decode_head=dict(
        induction_weights=[
            (11, 11, 30),
            (12, 12, 30),
            (11, 1, -8),
            (11, 2, -8),
            (12, 1, -8),
            (12, 2, -8)]))
```

To visualize the impact of induced bias, modify the compensation values and run

```shell
python tools/test.py configs/compensation_head/induction_compensation_head_deeplabv3plus_r50-d8_368x368_80k_kittistep.py your_model.pth --eval mIoU
```

### Plot compensation values

To plot the compensation values, run the following command for a trained model (e.g. on KITTI-STEP):

```shell
python tools/plot_compensation_matrix.py configs/compensation_head/compensation_head_deeplabv3plus_r50-d8_368x368_80k_kittistep.py your_model.pth
```

### Plot certainty distribution

To plot the uncompensated certainty distribution, run the following command for a trained model (e.g. on KITTI-STEP):

```shell
python tools/plot_certainty_distribution.py configs/compensation_head/compensation_head_deeplabv3plus_r50-d8_368x368_80k_kittistep.py your_model.pth
```

### Uncertainty estimation

To estimate the uncertainty of a trained model for a complete dataset, run the following command for a trained model (e.g. on KITTI-STEP):

```shell
python tools/predict_uncertainty.py configs/compensation_head/compensation_head_deeplabv3plus_r50-d8_368x368_80k_kittistep.py your_model.pth
```

You can add the flags

- **--store-raw-uncertainty** to additionally store the raw uncertainty maps (Default is just a
  visualization stored in a jpg file)
- **--store-prediction**  to additionally store the network prediction
- **--store-ground-truth**  to additionally store the ground truth files

### Competitors

This section presents the implementation and usage of some competitors that we used for comparison in our paper.

#### Bayesian Neural Network

We implemented a Bayesian Neural network into MMSegmentation to compare it to our method.
Configuration files and more details to the model can be found in the folder **configs/bayesian_deeplabv3plus** and in
the [README.md](configs/bayesian_deeplabv3plus/README.md) file in this folder.

To run the training, run

```shell
python tools/dist_train.py configs/bayesian_deeplabv3plus/deeplabv3plus_r50-d8_368x368_80k_kittistep.py 8
```

Inference, evaluation, uncertainty estimation can be performed similar as described above.

#### Noise Adaption Layer: Simple and Complex noise modeling

We implemented the Noise Adatption Layer into MMSegmentation to compare it to our method.
Configuration files and more details to the model can be found in the folder **configs/nal_head** and in
the README file in this folder.

To run the training, run

```shell
python tools/dist_train.py configs/nal_head/nal_head_simple_deeplabv3plus_r50-d8_368x368_80k_kittistep.py 8
```

Inference, evaluation, uncertainty estimation can be performed similar as described above.

#### LogComp

To employ LogComp, you need to change some parameters in the config files of the above described compensation method.
The parameters **local_compensation**, **non_diagonal** and **symmetric** need to be deactivated:

```python
_base_ = '../deeplabv3plus/deeplabv3plus_r50-d8_512x512_80k_ade20k.py'

model = dict(
    decode_head=dict(
        type='CompensationHead',
        local_compensation=False,
        non_diagonal=False,
        symmetric=False,
        decode_head={{_base_.model.decode_head}},
    ),
)
```

### Pretrained models

We provide pretrained models for some above mentioned methods:

| Method                                    | Dataset    | Download                                                                                                                                                                   |
| ----------------------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Baseline DeepLabV3+                       | Cityscapes | [model](https://www.tnt.uni-hannover.de/de/project/MPT/data/CompensationLearningInSemanticSegmentation/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.zip)                   |
| Baseline DeepLabV3+                       | KITTI-STEP | [model](https://www.tnt.uni-hannover.de/de/project/MPT/data/CompensationLearningInSemanticSegmentation/deeplabv3plus_r50-d8_368x368_80k_kittistep.zip)                     |
| Baseline DeepLabV3+                       | ADE20k     | [model](https://www.tnt.uni-hannover.de/de/project/MPT/data/CompensationLearningInSemanticSegmentation/deeplabv3plus_r50-d8_512x512_80k_ade20k.zip)                        |
| Baseline DeepLabV3+                       | COCO-stuff | [model](https://www.tnt.uni-hannover.de/de/project/MPT/data/CompensationLearningInSemanticSegmentation/deeplabv3plus_r50-d8_512x512_80k_coco-stuff10k.zip)                 |
| Bayesian DeepLabV3+                       | Cityscapes | [model](https://www.tnt.uni-hannover.de/de/project/MPT/data/CompensationLearningInSemanticSegmentation/.zip)                                                               |
| Bayesian DeepLabV3+                       | KITTI-STEP | [model](https://www.tnt.uni-hannover.de/de/project/MPT/data/CompensationLearningInSemanticSegmentation/.zip)                                                               |
| Bayesian DeepLabV3+                       | ADE20k     | [model](https://www.tnt.uni-hannover.de/de/project/MPT/data/CompensationLearningInSemanticSegmentation/.zip)                                                               |
| Bayesian DeepLabV3+                       | COCO-stuff | [model](https://www.tnt.uni-hannover.de/de/project/MPT/data/CompensationLearningInSemanticSegmentation/.zip)                                                               |
| Compensation DeepLabV3+                   | Cityscapes | [model](https://www.tnt.uni-hannover.de/de/project/MPT/data/CompensationLearningInSemanticSegmentation/compensation_head_deeplabv3plus_r50-d8_512x1024_80k_cityscapes.zip) |
| Compensation DeepLabV3+                   | KITTI-STEP | [model](https://www.tnt.uni-hannover.de/de/project/MPT/data/CompensationLearningInSemanticSegmentation/.zip)                                                               |
| Compensation DeepLabV3+                   | ADE20k     | [model](https://www.tnt.uni-hannover.de/de/project/MPT/data/CompensationLearningInSemanticSegmentation/.zip)                                                               |
| Compensation DeepLabV3+                   | COCO-stuff | [model](https://www.tnt.uni-hannover.de/de/project/MPT/data/CompensationLearningInSemanticSegmentation/.zip)                                                               |
| Noise Adaption Layer (simple) DeepLabV3+  | Cityscapes | [model](https://www.tnt.uni-hannover.de/de/project/MPT/data/CompensationLearningInSemanticSegmentation/nal_head_simple_deeplabv3plus_r50-d8_512x1024_80k_cityscapes.zip)   |
| Noise Adaption Layer (simple) DeepLabV3+  | KITTI-STEP | [model](https://www.tnt.uni-hannover.de/de/project/MPT/data/CompensationLearningInSemanticSegmentation/nal_head_simple_deeplabv3plus_r50-d8_368x368_80k_kittistep.zip)     |
| Noise Adaption Layer (simple) DeepLabV3+  | ADE20k     | [model](https://www.tnt.uni-hannover.de/de/project/MPT/data/CompensationLearningInSemanticSegmentation/nal_head_simple_deeplabv3plus_r50-d8_512x512_80k_ade20k.zip)        |
| Noise Adaption Layer (simple) DeepLabV3+  | COCO-stuff | [model](https://www.tnt.uni-hannover.de/de/project/MPT/data/CompensationLearningInSemanticSegmentation/nal_head_simple_deeplabv3plus_r50-d8_512x512_80k_coco-stuff10k.zip) |
| Noise Adaption Layer (complex) DeepLabV3+ | Cityscapes | [model](https://www.tnt.uni-hannover.de/de/project/MPT/data/CompensationLearningInSemanticSegmentation/nal_head_complex_deeplabv3plus_r50-d8_512x1024_80k_cityscapes.zip)  |
| Noise Adaption Layer (complex) DeepLabV3+ | KITTI-STEP | [model](https://www.tnt.uni-hannover.de/de/project/MPT/data/CompensationLearningInSemanticSegmentation/nal_head_complex_deeplabv3plus_r50-d8_368x368_80k_kittistep.zip)    |

NOTE: The provided models achieve slightly different results than the ones reported in the paper.
This is due to the fact that we reported mean mIoU over multiple runs, while the provided models are only trained once.

## License

This repository is developed upon [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). Both codebases are released under Apache 2.0 License themselves. Please check the MMSegmentation repo for details.
