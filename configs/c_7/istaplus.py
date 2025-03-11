_base_ = '_base_/img_dataset.py'

Phi_data_Name = '/data1/dym/GrokCSO-Dev/data/sampling_matrix/phi_7.mat'
Qinit_Name = '/data1/dym/GrokCSO-Dev/data/initial_matrix/Q_7.mat'

model = dict(
  type="ISTANetplus",
  LayerNo=6,
  Phi_data_Name=Phi_data_Name,
  Qinit_Name=Qinit_Name,
  c=7
)

