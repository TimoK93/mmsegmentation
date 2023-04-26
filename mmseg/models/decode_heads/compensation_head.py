# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

from ..builder import HEADS
from .loss_correction_head import BaseLossCorrectionHead


class NonDiagonal(nn.Module):

    @staticmethod
    def forward(X):
        y = X[:, :, 0, 0] * (1 - torch.eye(X.shape[0], device=X.device))
        return y[:, :, None, None]


class NonDiagonalAndSymetric(nn.Module):

    @staticmethod
    def forward(X):
        y = X[:, :, 0, 0].triu(1) + X[:, :, 0, 0].triu(1).transpose(0, 1) \
            * (1 - torch.eye(X.shape[0], device=X.device))
        return y[:, :, None, None]


class Induction(nn.Module):

    def __init__(self, induction_weights: list):
        super(Induction, self).__init__()
        self.induction_weights = induction_weights
        self.induction_matrix = None

    def forward(self, X):
        if self.induction_matrix is None:
            self.induction_matrix = torch.ones_like(X)
            for i, j, v in self.induction_weights:
                self.induction_matrix[j, i, 0, 0] = v
        y = X[:, :] + self.induction_matrix.to(X.device)
        return y


@HEADS.register_module()
class CompensationHead(BaseLossCorrectionHead):
    """
    Args:
        decode_head: Inherited from NoiseAdaptionLayerHead.
        non_diagonal (bool): Constrain elements in the compensation matrix
            to be zero at the diagonal. Default: True
        init_cfg: Inherited from BaseLossCorrectionHead.
    """

    def __init__(self,
                 decode_head,
                 non_diagonal=True,
                 symmetric=True,
                 local_compensation=True,
                 loss_balancing=1,
                 top_k=5,
                 induction_weights: list = None,
                 init_cfg=None,
                 **kwargs):
        super(CompensationHead, self).__init__(decode_head, init_cfg)

        self.loss_balancing = loss_balancing
        self.local_compensation = local_compensation
        self.non_diagonal = non_diagonal
        self.symmetric = symmetric
        self.top_k = top_k
        self.induction_weights = induction_weights

        self.compensations = nn.Conv2d(
            self.num_classes, self.num_classes, (1, 1), (1, 1), bias=False)
        self.compensations.weight.data = \
            torch.zeros_like(self.compensations.weight.data)

        if induction_weights is not None:
            parametrize.register_parametrization(
                self.compensations, 'weight', Induction(induction_weights))  #
        elif self.symmetric:
            assert self.non_diagonal
            parametrize.register_parametrization(self.compensations, 'weight',
                                                 NonDiagonalAndSymetric())
        elif non_diagonal:
            parametrize.register_parametrization(self.compensations, 'weight',
                                                 NonDiagonal())

        if self.local_compensation:
            in_channels = self.decode_head.in_channels if \
                type(self.decode_head.in_channels) == int else \
                self.decode_head.in_channels[-1]
            self.weight_layer = nn.Conv2d(
                in_channels, 64, (1, 1), (1, 1), bias=True)
            self.weight_layer.weight.data /= 10
            self.weight_layer.bias.data /= 10
            self.weight_layer2 = nn.Conv2d(64, 1, (1, 1), (1, 1), bias=False)
            self.weight_layer2.weight.data /= 10
            self.weight_layer_bn = torch.nn.BatchNorm2d(64)

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        # Calculate logits
        seg_logits = self.decode_head.forward(inputs)
        # Calculate weight
        ignore = gt_semantic_seg == self.decode_head.ignore_index
        compensation = \
            self._calc_compensation_and_one_hot_gt(gt_semantic_seg, ignore)
        weights = self._calc_weights(inputs, compensation)
        # Add weights to logits
        seg_logits = F.interpolate(
            seg_logits,
            compensation.shape[2:4],
            mode='bilinear',
            align_corners=False)
        seg_logits = seg_logits + compensation * weights
        # Calculate losses
        losses = self.losses(seg_logits, gt_semantic_seg)
        losses['loss_l1'] = self._l1_loss(compensation, ignore, weights)

        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Either standard or with induced bias.
        """
        output = self.decode_head.forward_test(inputs, img_metas, test_cfg)

        if self.induction_weights:
            probs = output.softmax(dim=1)
            compensation = self.compensations(probs)
            weights = self._calc_weights(inputs, compensation)
            compensation * weights
            output = output + compensation
        return output

    def _calc_compensation_and_one_hot_gt(self, gt_semantic_seg, ignore):
        gt = gt_semantic_seg.clone()
        gt[ignore] = 0
        gt = F.one_hot(gt, self.num_classes)
        gt = gt.permute(0, 4, 2, 3, 1).float()[:, :, :, :, 0]
        compensation = self.compensations(gt)
        compensation[:, :, ignore[0, 0]] = 0
        return compensation

    def _calc_weights(self, inputs, compensation):
        if self.local_compensation:
            weights = self.weight_layer(inputs[-1])
            weights = self.weight_layer_bn(weights)
            weights = self.weight_layer2(weights)
            weights = F.sigmoid(weights)
            weights = F.interpolate(
                weights,
                compensation.shape[2:4],
                mode='bilinear',
                align_corners=False)
        else:
            weights = 1

        return weights

    def _l1_loss(self, compensation, ignore, weight=1):
        l1 = compensation.abs().sum(
            dim=1, keepdim=True) / (
                self.num_classes - 1)
        l1 = self.loss_balancing * (l1.abs() *
                                    weight).sum() / (1 - 1 * ignore).sum()

        return l1

    def uncertainty(self, inputs, **kwargs):
        """Returns the uncertainty obtained by compensation."""
        # Calculate uncompensated probabilities
        seg_logits = self.decode_head.forward(inputs)
        probs_uncomp = seg_logits.softmax(dim=1)
        pseudo_gt = seg_logits.argmax(dim=1, keepdims=True)
        # Calculate weight
        ignore = pseudo_gt == -1  # Ignore is not existing in inference!
        compensation = self._calc_compensation_and_one_hot_gt(
            pseudo_gt, ignore)
        weights = self._calc_weights(inputs, compensation)
        # Initial probability of class
        guess = probs_uncomp == probs_uncomp.max(dim=1, keepdim=True)[0]
        initial_certainty = (probs_uncomp * guess).max(dim=1, keepdim=True)[0]
        # Iterate over top k classes for of compensation to get uncertainty
        mean_diff = torch.zeros_like(initial_certainty)
        probs = probs_uncomp.clone()
        for i in range(self.top_k):
            current_value, current_guess = probs.max(dim=1, keepdim=True)
            probs[probs == current_value] = 0
            pseudo_compensation = \
                self._calc_compensation_and_one_hot_gt(current_guess, ignore)
            comp = seg_logits + (pseudo_compensation * weights)
            p = (comp.softmax(dim=1) * guess).max(dim=1, keepdims=True)[0]
            mean_diff += (initial_certainty - p).pow(2) / self.top_k
        diff_uncertainty = mean_diff / mean_diff.max()
        uncertainty = diff_uncertainty * (1 - initial_certainty)

        return uncertainty
