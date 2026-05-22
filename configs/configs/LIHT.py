_base_ = '../_base_/datasets/DeRefNet.py'

env_cfg = dict(
    cudnn_benchmark=True
)

# phi和Q的路径
# phi表示以位置集导向矢量为列的矩阵，是一个121*1089的矩阵
Phi_data_Name = '/opt/data/private/Simon/DeRefNet/data/phi_0.5.mat'
# Q表示初始化矩阵，是一个1089*121的矩阵
Qinit_Name = '/opt/data/private/Simon/DeRefNet/data/track_5000_20/train/qinit.mat'


# 模型配置
model = dict(
  type="LIHT",  # 模型类型 全动态网络框架
  LayerNo=16,  # 层数，表示基本迭代模块的个数
  Phi_data_Name=Phi_data_Name,  # phi的路径
  Qinit_Name=Qinit_Name,  # Q的路径
)


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam', lr=0.0001, weight_decay=1e-5)
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
#                      image_name="DeRefNet")
# )
default_hooks = dict(
  checkpoint=dict(
    type='CheckpointHook',
    interval=-1,
    _scope_='mmdet',
    save_best='auto'))