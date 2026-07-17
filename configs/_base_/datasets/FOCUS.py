train_dataset_type = 'TrainDataset_Phix'
dataset_type = 'Val_Test_Dataset'

# 训练集位置
train_data_root = 'data/single_sigma/train/image'
val_data_root = 'data/single_sigma/val/image'
test_data_root = 'data/single_sigma/test/image'
val_xml_root = 'data/single_sigma/val/annotation'
test_xml_root = 'data/single_sigma/test/annotation'
train_xml_root = 'data/single_sigma/train/annotation'

train_dataloader = dict(
  batch_size=64,
  num_workers=2,
  pin_memory=True,
  sampler=dict(type='DefaultSampler', shuffle=True),
  dataset=dict(
    type=train_dataset_type,
    data_root=train_data_root,
    length=80000,
    xml_root = train_xml_root,
  )
)

val_dataloader = dict(
  batch_size=64,
  num_workers=2,
  pin_memory=True,
  sampler=dict(type='DefaultSampler', shuffle=True),
  drop_last=True,
  dataset=dict(
    type=dataset_type,
    data_dir=val_data_root,
    length=10000,
    xml_root = val_xml_root
  )
)

test_dataloader = dict(
    batch_size=64, 
    num_workers=2,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    drop_last=True,
    dataset=dict(
        type=dataset_type,
        data_dir=test_data_root,
        length=10000,
        xml_root = test_xml_root
    )
)


train_cfg = dict(
    by_epoch=True,
    max_epochs=100,
    val_interval=10
    )

val_cfg = dict()
test_cfg = dict()