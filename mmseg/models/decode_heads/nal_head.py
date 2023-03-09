# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F

from ..builder import HEADS
from .loss_correction_head import BaseLossCorrectionHead


@HEADS.register_module()
class NoiseAdaptionLayerHead(BaseLossCorrectionHead):
    """Noise adaption layer (NAL) as presented by Jacob Goldberger and Ehud
    Ben-Reuven. The NAL aims to learn the noise distribution in the training
    data and thus is able to predict the clean distribution based on the
    bayesian rule.

    The learned noise distribution is stored in the weights of the Conv2 layer
    `self.noise_adaption_layer'.

    Note that the features for complex model are down-sampled to n=32 due to
    the intractability for n x n matrices for large features dimensions.

    This head is the implementation of
        `TRAINING DEEP NEURAL-NETWORKS USING A NOISEADAPTATION LAYER
    <https://openreview.net/pdf?id=H12GRgcxg>`_.

    Args:
        decode_head: Inherited from NoiseAdaptionLayerHead.
        model_type (str): Either simple_model or complex_model according to the
            variants presented by the authors.
        noise_initialization (float): Uniform distributed noise
        init_cfg: Inherited from NoiseAdaptionLayerHead.
    """

    def __init__(self,
                 decode_head,
                 model_type='simple_model',
                 noise_initialisation=0.2,
                 init_cfg=None,
                 **kwargs):
        super(NoiseAdaptionLayerHead, self).__init__(decode_head, init_cfg)
        assert model_type in ['simple_model', 'complex_model']
        self.model_type = model_type
        self.noise_initialisation = noise_initialisation
        if self.model_type == 'simple_model':
            self.noise_adaption_layer = nn.Conv2d(
                self.num_classes, self.num_classes, (1, 1), bias=False)
        elif self.model_type == 'complex_model':
            in_channels = self.decode_head.in_channels if \
                type(self.decode_head.in_channels) == int else \
                self.decode_head.in_channels[-1]
            self.downsampling_layer = nn.Conv2d(
                in_channels, 32, (1, 1), bias=True)
            self.regression_layer = nn.Conv2d(
                32, self.num_classes * self.num_classes, (1, 1), bias=False)
            self.noise_adaption_layer = nn.Conv2d(
                self.num_classes, self.num_classes, (1, 1), bias=False)

    def init_weights(self):
        """Initialize assumed noise distribution according to the authors
        recommendation."""
        self.noise_adaption_layer.weight.data = \
            torch.ones_like(self.noise_adaption_layer.weight.data) * \
            self.noise_initialisation / (self.num_classes - 1)
        self.noise_adaption_layer.weight.data[
            torch.eye(self.num_classes)[:, :, None, None] == 1] = \
            1 - self.noise_initialisation
        self.noise_adaption_layer.weight.data = \
            self.noise_adaption_layer.weight.data.log()

        if self.model_type == 'complex_model':
            self.regression_layer.weight.data = torch.zeros_like(
                self.regression_layer.weight.data)

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_logits = self.decode_head.forward(inputs)
        feature_relevance = F.softmax(seg_logits, dim=1)

        if self.model_type == 'simple_model':
            weight = self.noise_adaption_layer.weight
            weight = F.softmax(weight, dim=1)
            final_weight = torch.conv2d(feature_relevance, weight)
        else:  # c_model
            weight_features = self.downsampling_layer(inputs[-1])
            weight_features = F.interpolate(
                weight_features,
                feature_relevance.shape[2:4],
                mode='bilinear',
                align_corners=False)
            weight = self.regression_layer(weight_features)
            weight = weight.view(
                (weight.shape[0], self.num_classes, self.num_classes,
                 weight.shape[-2], weight.shape[-1]))
            weight_bias = self.noise_adaption_layer.weight
            weight = weight + weight_bias[None]
            weight = F.softmax(weight, dim=2)
            final_weight = weight * feature_relevance[:, None]
            final_weight = final_weight.sum(dim=2)
        seg_logits = final_weight.log()
        losses = self.losses(seg_logits, gt_semantic_seg)

        return losses
