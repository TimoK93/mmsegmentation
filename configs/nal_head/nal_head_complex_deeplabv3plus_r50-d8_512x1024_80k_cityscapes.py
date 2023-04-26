_base_ = '../deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py'

model = dict(
    decode_head=dict(
        type='NoiseAdaptionLayerHead',
        model_type='complex_model',
        decode_head={{_base_.model.decode_head}}), )
# Setup for 4 GPUs
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
