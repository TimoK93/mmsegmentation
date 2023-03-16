# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import BaseModule

from .. import builder


class BaseLossCorrectionHead(BaseModule):
    """Wrapper around decode heads, such that modification in the input or
    output can be applied agnostic to the decode head or backbone.

    Args:
        decode_head (dict): Configuration for the decode head that is wrapped
            by the loss correction.
        init_cfg (dict): Configuration for initialization.
    """

    def __init__(self, decode_head, init_cfg=None, **kwargs):
        super(BaseLossCorrectionHead, self).__init__(init_cfg=init_cfg)
        # Child Head
        self.decode_head = builder.build_head(decode_head)
        # Inherit decode head attributes
        for item in dir(self.decode_head):
            if item not in dir(self):
                self.__setattr__(item, getattr(self.decode_head, item))
        # Copy important functions
        self._init_inputs = self.decode_head._init_inputs
        self._transform_inputs = self.decode_head._transform_inputs
        self.extra_repr = self.decode_head.extra_repr
        self.cls_seg = self.decode_head.cls_seg
        self.losses = self.decode_head.losses

    def forward(self, inputs):
        """Forward function wrapper.

        Overwrite if necessary.
        """
        output = self.decode_head.forward(inputs)
        return output

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function wrapper for training.

        Overwrite if necessary.
        """

        seg_logits = self.decode_head.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function wrapper for testing.

        Overwrite if necessary.
        """
        return self.decode_head.forward_test(inputs, img_metas, test_cfg)

    def uncertainty(self, inputs, **kwargs):
        """Returns the uncertainty obtained by compensation.

        Overwrite if necessary.
        """
        return self.decode_head.uncertainty(inputs, **kwargs)
