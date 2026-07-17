_base_ = '../_base_/datasets/FOCUS.py'

env_cfg = dict(
    cudnn_benchmark=True  # 设置为 True 以加速卷积操作
)

model = dict(
  type='FOCUS'
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam', lr=0.0003, weight_decay=1e-5)
    )

val_evaluator = dict(
  type="CSO_Metrics"
)
test_evaluator = dict(
  type="CSO_Metrics"
)

train_cfg = dict(
    by_epoch=True,
    max_epochs=300,
    val_interval=10
    )
val_cfg = dict()
test_cfg = dict()

# default_hooks = dict(
#   visualization=dict(type='CSOVisualizationHook', draw=True, c=3,
#                      image_name="SOURCE1")
# )
# 改 utils 中 show_contrast 函数内对应模型名称


default_hooks = dict(
  checkpoint=dict(
    type='CheckpointHook',
    interval=-1,
    _scope_='mmdet',
    save_best='auto'))
