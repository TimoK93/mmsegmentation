_base_ = '../segformer/segformer_mit-b5_512x512_160k_coco-stuff10k.py'

model = dict(
    decode_head=dict(
        type='NoiseAdaptionLayerHead',
        model_type='simple_model',
        decode_head={{_base_.model.decode_head}}), )
