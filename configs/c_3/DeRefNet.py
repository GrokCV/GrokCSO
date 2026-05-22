_base_ = '../_base_/datasets/DeRefNet.py'

env_cfg = dict(
    cudnn_benchmark=True
)

model = dict(
  type='DeRefNet',
  LayerNo=9
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
#                      image_name="DeRefNet")
# )
default_hooks = dict(
  checkpoint=dict(
    type='CheckpointHook',
    interval=-1,
    _scope_='mmdet',
    save_best='auto'))
