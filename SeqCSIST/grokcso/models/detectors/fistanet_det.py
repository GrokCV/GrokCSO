import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmagic.registry import MODELS
from mmengine.registry import MODELS
import scipy.io as sio
from tools.utils import read_targets_from_xml_list
from grokcso.models.backbones.fistanet_bb import initialize_weights, Fista_BasicBlock, l1_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@MODELS.register_module()
class FISTANet(BaseModel):
    def __init__(self,
                 LayerNo,
                 ):
        super(FISTANet, self).__init__()

        self.LayerNo = LayerNo

        Phi_data_Name = 'SeqCSIST/data/phi_0.5.mat'
        Phi_data = sio.loadmat(Phi_data_Name)
        Phi = Phi_data['phi']
        self.Phi = torch.from_numpy(Phi).type(torch.FloatTensor).to(device)

        Qinit_Name = 'SeqCSIST/data/track_5000_20/train/qinit.mat'
        Qinit_data = sio.loadmat(Qinit_Name)
        Qinit = Qinit_data['Qinit']
        self.Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor).to(device)

        onelayer = []

        self.bb = Fista_BasicBlock()
        for i in range(LayerNo):
            onelayer.append(self.bb)

        self.fcs = nn.ModuleList(onelayer)
        self.fcs.apply(initialize_weights)

        # thresholding value
        self.w_theta = nn.Parameter(torch.Tensor([-0.5]))
        self.b_theta = nn.Parameter(torch.Tensor([-2]))
        # gradient step
        self.w_mu = nn.Parameter(torch.Tensor([-0.2]))
        self.b_mu = nn.Parameter(torch.Tensor([0.1]))
        # two-step update weight
        self.w_rho = nn.Parameter(torch.Tensor([0.5]))
        self.b_rho = nn.Parameter(torch.Tensor([0]))

        self.Sp = nn.Softplus()

    def forward(self, **kwargs):
        mode = kwargs['mode']
        Phi = self.Phi
        Qinit = self.Qinit
        if mode == 'loss':
            # batch_x = torch.stack(lrs['matrices'])
            # Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))
            batch = torch.stack(kwargs["batch_x"])
            batch_x = batch.squeeze(dim=1)
            # batch_x = batch_x[2:12, :]
            Phi_x = torch.stack(kwargs["gt_img_11"])
            Phix = Phi_x.squeeze(dim=1)
            # Phix = Phix[2:12, :]

        elif mode == 'predict':
            image_name = []
            Phi_x = torch.stack(kwargs["gt_img_11"])
            Phix = Phi_x.squeeze(dim=1)
            # Phix = Phix[2:12, :]
            ann_paths = kwargs["ann_path"]
            targets_GT = read_targets_from_xml_list(ann_paths)
            image_name = kwargs["image_name"]
            
        else:
            print("Invalid mode:", mode)
            return None

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

        xold = x
        y = xold
        layers_sym = []  # for computing symmetric loss
        layers_st = []

        for i in range(self.LayerNo):
            theta_ = self.w_theta * i + self.b_theta
            mu_ = self.w_mu * i + self.b_mu

            [xnew, layer_sym, layer_st] = self.fcs[i](y, PhiTPhi, PhiTb, mu_, theta_)

            rho_ = (self.Sp(self.w_rho * i + self.b_rho) - self.Sp(
            self.b_rho)) / self.Sp(self.w_rho * i + self.b_rho)
            y = xnew + rho_ * (xnew - xold)  # two-step update
            xold = xnew

            layers_st.append(layer_st)
            layers_sym.append(layer_sym)

        x_final = xnew

        if mode == 'tensor':
            return x_final
        elif mode == 'predict':
            return [x_final[2:18, :], image_name, targets_GT]
        elif mode == 'loss':
            loss_discrepancy = torch.mean(torch.pow(x_final[2:18, :] - batch_x[2:18, :], 2)) + \
                            l1_loss(x_final[2:18, :], batch_x[2:18, :], 0.1)
            loss_constraint = 0
            for k, _ in enumerate(layers_sym, 0):
                loss_constraint += torch.mean(torch.pow(layers_sym[k], 2))

            sparsity_constraint = 0
            for k, _ in enumerate(layers_st, 0):
                sparsity_constraint += torch.mean(torch.abs(layers_st[k]))

            # loss = loss_discrepancy + gamma * loss_constraint
            loss = loss_discrepancy + 0.01 * loss_constraint + 0.001 * sparsity_constraint
            return {'loss': loss,
                    'loss_discrepancy':loss_discrepancy,
                    'loss_constraint':loss_constraint,
                    'sparsity_constraint':sparsity_constraint
                    }