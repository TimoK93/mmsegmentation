_base_ = '../segformer/segformer_mit-b5_368x368_160k_kittistep.py'

model = dict(
    decode_head=dict(
        type='NoiseAdaptionLayerHead',
        model_type='simple_model',
        decode_head={{_base_.model.decode_head}}), )
