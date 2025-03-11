_base_ = '../_base_/datasets/img_dataset.py'
Phi_data_Name = 'phi_3.mat'
Qinit_Name = 'Q_3.mat'


model = dict(
  type="LAMP",
  LayerNo=16,
  Phi_data_Name=Phi_data_Name,
  Qinit_Name=Qinit_Name,
  lam=0.4
)

