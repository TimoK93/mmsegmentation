_base_ = './compensation_head_deeplabv3plus_r50-d8_368x368_80k_kittistep.py'

model = dict(
    decode_head=dict(induction_weights=[
        (11, 11, 30),  # Push person with compensation
        (12, 12, 30),  # Push rider with compensation
        (11, 1, -8),  # Strengthen person against sidewalk
        (11, 2, -8),  # Strengthen person against building
        (12, 1, -8),  # Strengthen rider against sidewalk
        (12, 2, -8)
    ]))  # Strengthen rider against building
