_base_ = '../segformer/segformer_mit-b5_512x512_160k_ade20k.py'

model = dict(
    decode_head=dict(
        type='NoiseAdaptionLayerHead',
        model_type='simple_model',
        decode_head={{_base_.model.decode_head}}), )
evaluation = dict(interval=160000)
# Setup for 4 GPUs
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
