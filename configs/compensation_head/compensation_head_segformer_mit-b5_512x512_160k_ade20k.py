_base_ = '../segformer/segformer_mit-b5_512x512_160k_ade20k.py'

model = dict(
    decode_head=dict(
        type='CompensationHead',
        local_compensation=True,
        loss_balancing=2,
        non_diagonal=True,
        symmetric=True,
        top_k=5,
        decode_head={{_base_.model.decode_head}},
    ), )

evaluation = dict(interval=160000)
# Setup for 4 GPUs
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
