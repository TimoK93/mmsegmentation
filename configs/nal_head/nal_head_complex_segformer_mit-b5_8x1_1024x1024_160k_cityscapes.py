_base_ = '../segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes.py'

model = dict(
    decode_head=dict(
        type='NoiseAdaptionLayerHead',
        model_type='complex_model',
        decode_head={{_base_.model.decode_head}}), )
