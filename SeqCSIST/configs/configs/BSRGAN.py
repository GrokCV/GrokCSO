_base_ = '../_base_/datasets/DeRefNet.py'



env_cfg = dict(
    cudnn_benchmark=True
)

model = dict(
  type='BSRGAN',
  generator=dict(
        type='RRDBNet',
        in_channels=1,
        out_channels=1,
        mid_channels=64,
        num_blocks=16,
        upscale_factor=3),
  discriminator=dict(type='Discriminator_UNet', in_channels=1, mid_channels=64),
  pixel_loss=dict(type='L1Loss', loss_weight=1e-2, reduction='mean'),
  perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={'34': 1.0},
        vgg_type='vgg19',
        perceptual_weight=1.0,
        style_weight=0,
        norm_img=False),
  gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=5e-3,
        real_label_val=1.0,
        fake_label_val=0),
)

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(
#         type='Adam', lr=0.0001, weight_decay=1e-5)
#     )
# optimizer

optim_wrapper = dict(
    # _delete_=True,
    constructor='MultiOptimWrapperConstructor',
    generator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999))),
    discriminator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999))),
)

val_evaluator = dict(
  type="CSO_Metrics",
)
test_evaluator = dict(
  type="CSO_Metrics"
)

train_cfg = dict(
    by_epoch=True,
    max_epochs=10000,
    val_interval=100
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