_base_ = '../_base_/datasets/img_dataset.py'

Phi_data_Name = 'phi_3.mat'  # replace with the path to a_phi_0_3.mat
Qinit_Name = 'Q_3.mat'  # replace with the path to Q_3.mat

model = dict(
    type='DISTA_pp',
    LayerNo=6,
    Phi_data_Name = Phi_data_Name,  
    Qinit_Name = Qinit_Name,
    lambda_weight=0.7, 

    use_count=True,
    use_dtg=True,
    use_dt=True,
    use_offset=True,
    count_inside_dtg=True,
    ini_conv_in_offset=True,
    num_classes=5, # the range of target number
    
    offset_weight=300.0,
    count_weight=250.0,
    constraint_weight=0.01,

    c=3,
    nummodel='NumCNN2')

test_evaluator = [
  dict(
    type="CSO_Metrics",
    brightness_threshold=50,
    c=3,
    postprocess=True,
    ),
  dict(
    type="Similarity",
    c=3,
    brightness_threshold=50,
    postprocess=True,
  )
]
