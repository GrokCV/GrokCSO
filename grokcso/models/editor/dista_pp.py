import torch
from mmengine.registry import MODELS
import scipy.io as sio
from mmengine.model import BaseModel
from typing import Dict, Union
from tools.utils import read_targets_from_xml_list
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DTG_CountMod(nn.Module):
    def __init__(self, dim, in_features=64):
        super().__init__()
        self.count_mlp = nn.Sequential(
            nn.Linear(dim + in_features, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, dim),
        )
        self.convl = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv0_s = nn.Conv2d(dim, dim // 2, 1)
        self.conv1_s = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 3, padding=1)
        self.conv_m = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x, count_condition):
        batch_size = count_condition.shape[0]
        x_pooled = x.mean(dim=[2, 3])
        x_pooled += x.amax(dim=[2, 3])
        fused_condition = torch.cat([x_pooled, count_condition], dim=1)
        cm = self.count_mlp(fused_condition).view(batch_size, -1, 1, 1)
        x_count = x + x * torch.sigmoid(cm)

        attn2 = self.convl(x)
        attn1 = self.conv0_s(x)
        attn2 = self.conv1_s(attn2)
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = (attn1 * sig[:, 0, :, :].unsqueeze(1)
                + attn2 * sig[:, 1, :, :].unsqueeze(1))
        attn = self.conv_m(attn)
        return x_count * attn


class DynamicSpatialAttention(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.kernel_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, kernel_size ** 2, kernel_size=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, initial_x, x_input):
        batch_size, _, height, width = initial_x.shape
        kernels = self.kernel_generator(initial_x).view(
            batch_size, 1, self.kernel_size, self.kernel_size,
        )
        x_input = x_input.view(1, batch_size, height, width)
        att = F.conv2d(
            x_input, weight=kernels,
            padding=self.kernel_size // 2, groups=batch_size,
        )
        att = att.view(batch_size, 1, height, width)
        return self.sigmoid(att)


class DIST_DT_DTG_CountInside_Block(nn.Module):
    def __init__(self, lambda_weight, c):
        super().__init__()
        self.c = c
        self.channels = 64
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 1, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 64, 3, 3)))
        self.dynamic_attention = DynamicSpatialAttention(in_channels=1, kernel_size=3)
        self.lambda_weight = torch.Tensor([lambda_weight]).to(device)
        self.dtg = DTG_CountMod(64)
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 64, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 64, 3, 3)))

    def forward(self, x, count_scale, PhiTPhi, PhiTb):
        initial_x = x
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 1, 11 * self.c, 11 * self.c)

        x_s = F.conv2d(x_input, self.conv1_forward, padding=1)
        x_s = F.relu(x_s)
        x_static = F.conv2d(x_s, self.conv2_forward, padding=1)
        x_dynamic = self.dynamic_attention(
            initial_x.view(-1, 1, 11 * self.c, 11 * self.c), x_input,
        )
        x_forward = self.lambda_weight * x_static + (1 - self.lambda_weight) * x_dynamic
        dt = self.dtg(x_forward, count_scale)
        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - dt))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)
        x_pred = x_backward.view(-1, 11 * self.c * 11 * self.c)

        x_sym = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x_sym = F.relu(x_sym)
        x_est = F.conv2d(x_sym, self.conv2_backward, padding=1)
        symloss = x_est - x_input
        return [x_pred, symloss, dt]


class INI_ConvOnly(torch.nn.Module):
    def __init__(self, out_dim, in_dim):
        super(INI_ConvOnly, self).__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)

        return x


class ScaleTanh(nn.Module):
    def __init__(self, scale=0.5):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return self.scale * torch.tanh(x)


class Offset(torch.nn.Module):
    def __init__(self, out_dim, in_dim, act='tanh'):
        super(Offset, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, out_dim * 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim * 16, out_dim, kernel_size=3, padding=1),
        )
        if act == 'tanh':
            self.act_out = nn.Tanh()
        elif act == 'scaled_tanh':
            self.act_out = ScaleTanh(scale=0.5)  # 0.5 * tanh(x)
        elif act == 'none':
            self.act_out = nn.Identity()
        else:
            raise ValueError(f"Unknown act for Offset: {act!r}")

    def forward(self, x):
        return self.act_out(self.net(x))


class NumCNN2(torch.nn.Module):
    def __init__(self, num_classes=5):
        super(NumCNN2, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()
        self.fc_mid = nn.Sequential(
            nn.Linear(64 * 2 * 2, 64),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(-1, 1, 11, 11)
        x = self.conv_layers(x)
        x = self.flatten(x)

        mid_feat = self.fc_mid(x)
        count_logits = self.fc_out(self.dropout(mid_feat))

        return count_logits, mid_feat


@MODELS.register_module()
class DISTA_pp(BaseModel):
    def __init__(self,
                 LayerNo,
                 Phi_data_Name,
                 Qinit_Name,
                 lambda_weight=0.5,
                 c=3,
                 is_mat=False,
                 nummodel="NumCNN2",
                 offset_weight=1.0,
                 count_weight=1.0,
                 constraint_weight=0.01,
                 use_count=True,
                 use_dtg=True,
                 use_dt=True,
                 use_offset=True,
                 ini_conv_in_offset=True,
                 count_inside_dtg=True,
                 num_classes=5,
                 offset_act='tanh',
                 ):
        super(DISTA_pp, self).__init__()

        required_options = dict(
            use_count=use_count, use_dtg=use_dtg, use_dt=use_dt,
            use_offset=use_offset, ini_conv_in_offset=ini_conv_in_offset,
            count_inside_dtg=count_inside_dtg,
        )
        disabled = [name for name, enabled in required_options.items() if not enabled]
        if disabled:
            raise ValueError(
                'DISTA_pp implements the full DT/DTG/count/offset architecture; '
                f'these options must be True: {", ".join(disabled)}'
            )

        self.LayerNo = LayerNo
        self.c = c
        self.is_mat = is_mat
        self.offset_weight = offset_weight
        self.count_weight = count_weight
        self.constraint_weight = constraint_weight

        self.num_classes = num_classes
        self.offset_act = offset_act

        Phi_data = sio.loadmat(Phi_data_Name)
        Phi = Phi_data['phi']
        self.Phi = torch.from_numpy(Phi).type(torch.FloatTensor).to(device)

        Qinit_data = sio.loadmat(Qinit_Name)
        Qinit = Qinit_data['Qinit']
        self.Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor).to(device)

        self.numcnn = eval(nummodel)(num_classes=num_classes)
        self.criteria = nn.CrossEntropyLoss()
        self.init_z_conv_only = INI_ConvOnly(out_dim=64, in_dim=1)
        self.fcs = nn.ModuleList([
            DIST_DT_DTG_CountInside_Block(lambda_weight=lambda_weight, c=c)
            for _ in range(LayerNo)
        ])
        self.offset_head = Offset(out_dim=2, in_dim=65, act=self.offset_act)

    def forward(self, **kwargs) -> Union[Dict[str, torch.Tensor], list]:
        mode = kwargs['mode']
        Phi = self.Phi
        Qinit = self.Qinit

        if mode == 'tensor':
            Phix = kwargs["Phix"]
        else:
            Phix = torch.stack(kwargs["gt_img_11"]).squeeze(dim=1).to(device)
            Input_image = torch.stack(kwargs["gt_img_11"])
            batch_x = torch.stack(kwargs["gt"]).squeeze(dim=1).to(device)
            ann_paths = kwargs["ann_path"]
            file_id = kwargs["file_id"]
            counts = kwargs["count_1"]
            counts = torch.tensor(counts, dtype=torch.int64).to(device)

        external_count_condition = kwargs.get('external_count_condition', None)
        count_pred_idx = None
        if external_count_condition is None or mode == 'loss':
            count_logits, count_condition = self.numcnn(Phix)
        if external_count_condition is not None:
            count_condition = external_count_condition

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        if Phix.dim() > 2:
            Phix = Phix.reshape(-1, Phix.shape[-1])
        PhiTb = torch.mm(Phix, Phi)
        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

        x_input = x.view(-1, 1, 11 * self.c, 11 * self.c)
        ini_conv_feat = self.init_z_conv_only(x_input)

        layers_sym = []
        dt_list = []
        intermediate = [x]

        for i in range(self.LayerNo):
            x, layer_sym, dt = self.fcs[i](x, count_condition, PhiTPhi, PhiTb)

            intermediate.append(x)
            layers_sym.append(layer_sym)
            dt_list.append(dt)

        x_final = x

        x_final_img = x_final.view(-1, 1, 11 * self.c, 11 * self.c)
        feat_combined = torch.cat([x_final_img, ini_conv_feat], dim=1)
        offset_pred = self.offset_head(feat_combined)

        if mode == 'tensor':
            return x_final, count_condition, intermediate, count_pred_idx, dt_list

        elif mode == 'predict':
            targets_GT = read_targets_from_xml_list(ann_paths, self.is_mat)
            
            return [{"x_final": x_final,
                     "targets_GT": targets_GT,
                     "file_id": file_id,
                     "Input_image": Input_image,
                     "img_gt": batch_x,
                     "count": counts,
                     "x_num": count_condition,
                     "count_pred_idx": count_pred_idx,
                     "offset_pred": offset_pred,
                     "intermediate": intermediate,
                     "dt_list": dt_list,
                     }]

        elif mode == 'loss':
            loss_discrepancy = torch.mean(torch.pow(x_final - batch_x, 2))
            loss_constraint = torch.mean(torch.pow(layers_sym[0], 2))
            for k in range(self.LayerNo - 1):
                loss_constraint += torch.mean(torch.pow(layers_sym[k + 1], 2))
            loss_count = self.criteria(count_logits, counts) * self.count_weight

            offset_gt = torch.stack(kwargs["offset_gt"]).squeeze(dim=1).to(device)
            offset_mask = torch.stack(kwargs["offset_mask"]).squeeze(dim=1).to(device)
            offset_mask = offset_mask.unsqueeze(1)
            loss_offset = F.l1_loss(offset_pred * offset_mask, offset_gt * offset_mask, reduction='sum')
            loss_offset = loss_offset / (offset_mask.sum() + 1e-4) * self.offset_weight

            return {'loss_discrepancy': loss_discrepancy,
                    'loss_count': loss_count,
                    'loss_constraint': loss_constraint * self.constraint_weight,
                    'loss_offset': loss_offset}
