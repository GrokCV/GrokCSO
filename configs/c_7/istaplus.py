_base_ = '_base_/img_dataset.py'

Phi_data_Name = 'phi_7.mat'  # replace with the path to phi_7.mat
Qinit_Name = 'Q_7.mat'  # replace with the path to Q_7.mat

model = dict(
  type="ISTANetplus",
  LayerNo=6,
  Phi_data_Name=Phi_data_Name,
  Qinit_Name=Qinit_Name,
  c=7
)

