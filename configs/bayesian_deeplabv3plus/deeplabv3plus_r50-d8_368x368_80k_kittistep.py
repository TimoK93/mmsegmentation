_base_ = '../deeplabv3plus/deeplabv3plus_r50-d8_368x368_80k_kittistep.py'
model = dict(
    type='BayesianEncoderDecoder',
    iterations=10,
    backbone=dict(dropout_rates=(None, None, 0.25, None)))
