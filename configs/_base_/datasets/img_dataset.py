"""
  用于定义数据集的配置文件1
"""

# 数据集类型
dataset_type = 'NoiseDataset'

# 训练集位置
train_data_root = '/data1/dym/GrokCSO-Dev/data/cso_data/train'  # 训练集位置
val_data_root = '/data1/dym/GrokCSO-Dev/data/cso_data/val'  # 验证集位置
test_data_root = '/data1/dym/GrokCSO-Dev/data/cso_data/test'  # 测试集位置

# 训练集集加载器
train_dataloader = dict(
  batch_size=64,  # 批次大小
  num_workers=2,  # 工作线程数
  pin_memory=True,  # 是否将数据保存在固定内存中
  sampler=dict(type='DefaultSampler', shuffle=True),  # 采样器，类型为默认采样器，打乱数据
  dataset=dict(
    type=dataset_type,  # 数据集类型 TrainDataset
    data_root=train_data_root,   # 数据集位置
    length=80000  # 训练集大小
  )
)

# 验证集加载器
val_dataloader = dict(
  batch_size=64,  # 批次大小
  num_workers=2,  # 工作线程数
  pin_memory=True,  # 是否将数据保存在固定内存中
  sampler=dict(type='DefaultSampler', shuffle=True),  # 采样器，类型为默认采样器，打乱数据
  dataset=dict(
    type=dataset_type,  # 数据集类型 TestDataset
    data_root=val_data_root,  # 验证集位置
    length=10000  # 验证集大小统一为9984
  ),
  drop_last=True
)

# 测试集加载器
test_dataloader = dict(
  batch_size=64,  # 批次大小
  num_workers=2,  # 工作线程数
  pin_memory=True,  # 是否将数据保存在固定内存中
  sampler=dict(type='DefaultSampler', shuffle=False),  # 采样器，类型为默认采样器，不打乱数据
  dataset=dict(
    type=dataset_type,  # 数据集类型 TestDataset
    data_root=test_data_root,  # 测试集位置
    length=10000  # 测试集大小统一为9984
  ),
  drop_last=True
)

train_cfg = dict(
    by_epoch=True,  # 按照epoch训练
    max_epochs=250,  # 最大训练次数
    val_interval=1  # 验证间隔为1
    )

val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam', lr=0.0001),  # 优化器类型为Adam，学习率为0.0001
    )

val_evaluator = [
  dict(
    type="CSO_Metrics",
    brightness_threshold=50,
    c=3)
]
test_evaluator = val_evaluator  # 测试评估器与验证评估器相同
