"""
ista-net模型的配置文件
  Description:
    detector: ISTA
    basic_block: BasicBlock
"""

_base_ = '../_base_/datasets/img_dataset.py'  # 数据集配置文件使用img_dataset.py

# phi和Q的路径，
# phi表示以位置集导向矢量为列的矩阵，是一个121*1089的矩阵
Phi_data_Name = '/data1/dym/GrokCSO-Dev/data/sampling_matrix/a_phi_0_3.mat'
# Q表示初始化矩阵，是一个1089*121的矩阵
Qinit_Name = '/data1/dym/GrokCSO-Dev/data/initial_matrix/Q_3.mat'

# 模型框架下具有相同结构的基本迭代模块 BasicBlock
block = "BasicBlock"

# 模型配置
model = dict(
  type="FDFrameWork",  # 模型类型
  LayerNo=9,  # 层数，表示基本迭代模块的个数
  Phi_data_Name=Phi_data_Name,  # phi的路径
  Qinit_Name=Qinit_Name,  # Q的路径
  block=block  # 基本迭代模块
)
resume=True

