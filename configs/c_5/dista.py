_base_ = '_base_/img_dataset.py'

Phi_data_Name = 'phi_5.mat'
Q_init_Name = 'Q_5.mat'
block = "DIST_BasicBlock"

model = dict(
  type="FDFrameWork",
  LayerNo=6,
  Phi_data_Name=Phi_data_Name,
  Qinit_Name=Q_init_Name,
  block=block,
  lambda_weight=0.7,
  c=5
)


