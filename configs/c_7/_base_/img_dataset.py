_base_ = '../../_base_/datasets/img_dataset.py'
dataset_type = 'NoiseDataset'
train_data_root = 'train'

train_dataloader = dict(
  batch_size=64,
  num_workers=2,
  pin_memory=True,
  sampler=dict(type='DefaultSampler', shuffle=True),
  dataset=dict(
    type=dataset_type,
    data_root=train_data_root,
    length=80000,
    c=7
  )
)


val_evaluator = [
  dict(
    type="CSO_Metrics",
    brightness_threshold=50,
    c=7),
]
test_evaluator = val_evaluator
