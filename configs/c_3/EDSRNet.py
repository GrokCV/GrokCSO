_base_ = '../_base_/datasets/DeRefNet.py'

env_cfg = dict(
    cudnn_benchmark=True
)

model = dict(
  type='EDSRNet'
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam', lr=0.0001)
    )

val_evaluator = dict(
  type="CSO_Metrics",
)
test_evaluator = dict(
  type="CSO_Metrics"
)

train_cfg = dict(
    by_epoch=True,
    max_epochs=100,
    val_interval=10
    )
val_cfg = dict()
test_cfg = dict()

# default_hooks = dict(
#   visualization=dict(type='CSOVisualizationHook', draw=True, c=3,
#                      image_name="EDSR")
# )
default_hooks = dict(
  checkpoint=dict(
    type='CheckpointHook',
    interval=-1,
    _scope_='mmdet',
    save_best='auto'))
