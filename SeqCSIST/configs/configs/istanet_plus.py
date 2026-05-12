_base_ = '../_base_/datasets/DeRefNet.py'

env_cfg = dict(
    cudnn_benchmark=True  # 设置为 True 以加速卷积操作
)

model = dict(
  type='ISTANetplus',
  LayerNo=9
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam', lr=0.0001, weight_decay=1e-5)
    )

val_evaluator = dict(
  type="CSO_Metrics",
  # type="SINR_DET_RATE"
)
test_evaluator = dict(
  type="CSO_Metrics"
  # type="SINR_DET_RATE"
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
#                      image_name="ISTANet")
# )

default_hooks = dict(
  checkpoint=dict(
    type='CheckpointHook',
    interval=-1,
    _scope_='mmdet',
    save_best='auto'))