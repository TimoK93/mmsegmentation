_base_ = '../deeplabv3plus/deeplabv3plus_r50-d8_512x512_80k_coco-stuff10k.py'

model = dict(
    decode_head=dict(
        type='NoiseAdaptionLayerHead',
        model_type='simple_model',
        decode_head={{_base_.model.decode_head}}), )
