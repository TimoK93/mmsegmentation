# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmseg.ops import resize
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class BayesianEncoderDecoder(EncoderDecoder):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self, iterations=10, **kwargs):
        super(BayesianEncoderDecoder, self).__init__(**kwargs)
        self.iterations = iterations

    def encode_decode(self, img, img_metas, return_variance=False):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        out = list()
        self.backbone.dropout(True)
        for i in range(self.iterations):
            x = self.extract_feat(img)
            seg_logit = self._decode_head_forward_test(x, img_metas)
            probabilities = F.softmax(seg_logit, dim=1)
            out.append(probabilities)
        self.backbone.dropout(False)
        mean = torch.zeros_like(out[0])
        variance = torch.zeros_like(out[0])
        for o in out:
            mean += o / self.iterations
        for o in out:
            variance += (o - mean).pow(2) / self.iterations
        out = mean.log()
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if return_variance:
            variance = resize(
                input=variance,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            variance = variance * (out == out.max(dim=1, keepdims=True)[0])
            variance = variance.sum(dim=1, keepdims=True)
            return out, variance
        return out

    def forward_with_uncertainty(self, img, **kwargs):
        """Dummy forward function."""
        seg_logit, var = self.encode_decode(img, None, return_variance=True)
        std_dev = var.sqrt()
        return seg_logit, std_dev

    def simple_test_logits_with_uncertainty(self,
                                            img,
                                            img_metas,
                                            rescale=True):
        """Test single image without augmentations with uncertainty.

        Return numpy seg_map logits and std deviation.
        """
        seg_logit, var = self.encode_decode(img[0], None, return_variance=True)
        std_dev = var.sqrt()
        std_dev = std_dev.cpu().numpy()
        seg_logit = seg_logit.cpu().numpy()
        return seg_logit, std_dev
