_base_ = '../segformer/segformer_mit-b5_368x368_160k_kittistep.py'

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
    samples_per_gpu=2,
    workers_per_gpu=2,
)
