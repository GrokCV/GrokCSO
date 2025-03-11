_base_ = '_base_/img_dataset.py'

Phi_data_Name = '/data1/dym/GrokCSO-Dev/data/sampling_matrix/phi_7.mat'
Q_init_Name = '/data1/dym/GrokCSO-Dev/data/initial_matrix/Q_7.mat'
block = "DIST_BasicBlock"

model = dict(
  type="FDFrameWork",
  LayerNo=4,
  Phi_data_Name=Phi_data_Name,
  Qinit_Name=Q_init_Name,
  block=block,
  lambda_weight=0.7,
  c=7
)
train_cfg = dict(
    by_epoch=True,
    max_epochs=250,
    val_interval=1
    )




