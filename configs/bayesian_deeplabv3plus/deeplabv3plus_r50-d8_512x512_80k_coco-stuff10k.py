_base_ = '../deeplabv3plus/deeplabv3plus_r50-d8_512x512_80k_coco-stuff10k.py'
model = dict(
    type='BayesianEncoderDecoder',
    iterations=10,
    backbone=dict(dropout_rates=(None, None, 0.25, None)))

evaluation = dict(interval=100)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
)
