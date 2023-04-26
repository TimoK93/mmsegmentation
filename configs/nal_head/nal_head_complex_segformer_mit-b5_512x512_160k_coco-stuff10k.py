_base_ = '../segformer/segformer_mit-b5_512x512_160k_coco-stuff10k.py'

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
