_base_ = '../../_base_/datasets/img_dataset.py'

dataset_type = 'NoiseDataset'

train_data_root = 'train'
val_data_root = 'val'
test_data_root = 'test'

train_dataloader = dict(
  batch_size=64,
  num_workers=2,
  pin_memory=True,
  sampler=dict(type='DefaultSampler', shuffle=True),
  dataset=dict(
    type=dataset_type,
    data_root=train_data_root,
    length=80000,
    c=5
  )
)

val_dataloader = dict(
  batch_size=64,
  num_workers=2,
  pin_memory=True,
  sampler=dict(type='DefaultSampler', shuffle=True),
  dataset=dict(
    type=dataset_type,
    data_root=val_data_root,
    length=10000,
    c=7
  ),
  drop_last=True
)

test_dataloader = dict(
  batch_size=64,
  num_workers=2,
  pin_memory=True,
  sampler=dict(type='DefaultSampler', shuffle=False),
  dataset=dict(
    type=dataset_type,
    data_root=test_data_root,
    length=10000,
    c=7
  ),
  drop_last=True
)


val_evaluator = [
  dict(
    type="CSO_Metrics",
    brightness_threshold=50,
    c=5),
  dict(
    type="Similarity",
    c=5),
]
test_evaluator = val_evaluator
