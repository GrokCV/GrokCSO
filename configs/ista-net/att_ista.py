_base_ = '../_base_/datasets/img_dataset.py'
Phi_data_Name = 'phi_3.mat'  # replace with the path to phi_3.mat
Qinit_Name = 'Q_3.mat'  # replace with the path to Q_3.mat

block = "att_BasicBlock"

model = dict(
  type="FDFrameWork",
  LayerNo=6,
  Phi_data_Name=Phi_data_Name,
  Qinit_Name=Qinit_Name,
  block=block,
)
