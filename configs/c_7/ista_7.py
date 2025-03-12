_base_ = '_base_/img_dataset.py'

Phi_data_Name = 'phi_7.mat'  # replace with the path to phi_7.mat
Q_init_Name = 'Q_7.mat'  # replace with the path to Q_7.mat

block = "BasicBlock"

model = dict(
  type="FDFrameWork",
  LayerNo=9,
  Phi_data_Name=Phi_data_Name,
  Qinit_Name=Q_init_Name,
  block=block,
  c=7
)

