_base_ = '../deeplabv3plus/deeplabv3plus_r50-d8_368x368_80k_kittistep.py'

model = dict(
    decode_head=dict(
        type='NoiseAdaptionLayerHead',
        model_type='complex_model',
        decode_head={{_base_.model.decode_head}}), )
