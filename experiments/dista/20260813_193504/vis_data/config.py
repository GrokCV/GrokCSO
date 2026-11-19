Phi_data_Name = '/defaultShare/archive/xumengze/GrokCSO/data/sampling_matrix/phi_3.mat'
Qinit_Name = '/defaultShare/archive/xumengze/GrokCSO/data/initial_matrix/Q_3.mat'
dataset_type = 'NoiseDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=1,
        rule='greater',
        save_best='cso_metric/mAP',
        type='CheckpointHook'))
launcher = 'none'
load_from = '/defaultShare/archive/xumengze/jiehun/MODEL/GrokCSOv2/work_dirs_tpami/dista/best_cso_metric_mAP_epoch_91.pth'
model = dict(
    LayerNo=6,
    Phi_data_Name=
    '/defaultShare/archive/xumengze/GrokCSO/data/sampling_matrix/phi_3.mat',
    Qinit_Name=
    '/defaultShare/archive/xumengze/GrokCSO/data/initial_matrix/Q_3.mat',
    c=3,
    count_in_offset=False,
    count_inside_dtg=False,
    lambda_weight=0.7,
    nummodel='NumCNN2',
    type='DISTA_pp',
    use_count=False,
    use_dt=True,
    use_dtg=True,
    use_offset=False)
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0, norm_type=2),
    optimizer=dict(lr=0.0001, type='Adam'),
    type='OptimWrapper')
test_cfg = dict()
test_data_root = '/defaultShare/archive/xumengze/datasets/cso_data/test'
test_dataloader = dict(
    batch_size=64,
    dataset=dict(
        data_root='/defaultShare/archive/xumengze/datasets/cso_data/test',
        length=10000,
        type='NoiseDataset'),
    drop_last=True,
    num_workers=2,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(brightness_threshold=50, c=3, type='CSO_Metrics'),
    dict(c=3, type='Similarity'),
]
train_cfg = dict(by_epoch=True, max_epochs=250, val_interval=1)
train_data_root = '/defaultShare/archive/xumengze/datasets/cso_data/train'
train_dataloader = dict(
    batch_size=64,
    dataset=dict(
        data_root='/defaultShare/archive/xumengze/datasets/cso_data/train',
        length=80000,
        type='NoiseDataset'),
    num_workers=2,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
val_cfg = dict()
val_data_root = '/defaultShare/archive/xumengze/datasets/cso_data/val'
val_dataloader = dict(
    batch_size=64,
    dataset=dict(
        data_root='/defaultShare/archive/xumengze/datasets/cso_data/val',
        length=10000,
        type='NoiseDataset'),
    drop_last=True,
    num_workers=2,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
val_evaluator = [
    dict(brightness_threshold=50, c=3, type='CSO_Metrics'),
    dict(c=3, type='Similarity'),
]
work_dir = '/defaultShare/archive/xumengze/jiehun/MODEL/GrokCSOv2/work_dirs_tpami/dista'
