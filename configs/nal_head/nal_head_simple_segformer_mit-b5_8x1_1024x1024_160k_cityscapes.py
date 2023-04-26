_base_ = '../segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes.py'

model = dict(
    decode_head=dict(
        type='NoiseAdaptionLayerHead',
        model_type='simple_model',
        decode_head={{_base_.model.decode_head}}), )
evaluation = dict(interval=160000)
# Setup for 4 GPUs
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
)
