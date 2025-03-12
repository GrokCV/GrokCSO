_base_ = '_base_/img_dataset.py'
Phi_data_Name = 'phi_5.mat'  # replace with the path to phi_5.mat
Qinit_Name = 'Q_5.mat'  # replace with the path to Q_5.mat


model = dict(
  type="ISTANetplus",
  LayerNo=6,
  Phi_data_Name=Phi_data_Name,
  Qinit_Name=Qinit_Name,
  c=5
)

