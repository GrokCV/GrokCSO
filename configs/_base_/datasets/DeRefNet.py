train_dataset_type = 'TrainDataset_Phix'
dataset_type = 'Val_Test_Dataset'

# 训练集位置
train_data_root = 'SeqCSIST/data/track_5000_20/train/image'
val_data_root = 'SeqCSIST/data/track_5000_20/val/image'
test_data_root = 'SeqCSIST/data/track_5000_20/test/image'
val_xml_root = 'SeqCSIST/data/track_5000_20/val/annotation'
test_xml_root = 'SeqCSIST/data/track_5000_20/test/annotation'
train_xml_root = 'SeqCSIST/data/track_5000_20/train/annotation'

train_dataloader = dict(
  batch_size=20,
  num_workers=2,
  pin_memory=True,
  # sampler=dict(type='DefaultSampler', shuffle=False),
  sampler=dict(type='ContinuousSampler', shuffle=False),
  dataset=dict(
    type=train_dataset_type,
    data_root=train_data_root,
    length=70000,
    # length=9800,
    xml_root = train_xml_root,
  )
)

val_dataloader = dict(
  batch_size=20,
  num_workers=2,
  pin_memory=True,
  sampler=dict(type='ContinuousSampler', shuffle=False),
  dataset=dict(
    type=dataset_type,
    data_dir=val_data_root,
    length=15000,
    # length=2100,
    xml_root = val_xml_root
  )
)

test_dataloader = dict(
  batch_size=20, 
  num_workers=2,
  pin_memory=True,
  sampler=dict(type='ContinuousSampler', shuffle=False),
  dataset=dict(
    type=dataset_type,
    data_dir=test_data_root,
    length=15000,
    # length=2100,
    xml_root = test_xml_root
  )
)

# test_dataloader = val_dataloader


train_cfg = dict(
    by_epoch=True,
    max_epochs=300,
    val_interval=10
    )

val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam', lr=0.0001, weight_decay=1e-5)
    )

val_evaluator = dict(
  type="CSO_Metrics"
)
test_evaluator = dict(
  type="CSO_Metrics"
)

# default_hooks = dict(
#   visualization=dict(type='CSOVisualizationHook', draw=True, c=3,
#                      image_name="SRCNN")
# )

default_hooks = dict(
  checkpoint=dict(
    type='CheckpointHook',
    interval=-1,
    _scope_='mmdet',
    save_best='auto'))