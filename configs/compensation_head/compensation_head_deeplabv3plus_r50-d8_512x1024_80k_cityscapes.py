_base_ = '../deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py'

model = dict(
    decode_head=dict(
        type='CompensationHead',
        local_compensation=True,
        loss_balancing=0.01,
        non_diagonal=True,
        symmetric=True,
        top_k=5,
        decode_head={{_base_.model.decode_head}},
    ), )

evaluation = dict(interval=80000)
# Setup for 4 GPUs
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
