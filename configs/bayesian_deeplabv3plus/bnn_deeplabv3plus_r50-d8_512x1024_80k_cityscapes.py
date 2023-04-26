_base_ = '../deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py'
model = dict(
    type='BayesianEncoderDecoder',
    iterations=10,
    backbone=dict(dropout_rates=(None, None, 0.25, None)))
evaluation = dict(interval=80000)
# Setup for 4 GPUs
# Setup for 4 GPUs
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
