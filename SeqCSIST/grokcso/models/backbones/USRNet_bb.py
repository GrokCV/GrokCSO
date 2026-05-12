import torch
import torch.nn as nn
from grokcso.models.backbones import basicblock as B
import numpy as np
from tools import util_image as util

# for pytorch version <= 1.7.1

"""
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
@inproceedings{zhang2020deep,
  title={Deep unfolding network for image super-resolution},
  author={Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={0--0},
  year={2020}
}
# --------------------------------------------
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
# --------------------------------------------
# basic functions
# --------------------------------------------
"""

class MSEloss(nn.Module):
    def __init__(self, alpha=0.9):
        super(MSEloss, self).__init__()
        self.MSE = nn.MSELoss()
        self.alpha = alpha
    def forward(self, img1, img2):
        loss = self.alpha*self.MSE(img1, img2)
        return loss


def splits(a, sf):
    '''split a into sfxsf distinct blocks

    Args:
        a: NxCxWxHx2
        sf: split factor

    Returns:
        b: NxCx(W/sf)x(H/sf)x2x(sf^2)
    '''
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=5)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=5)
    return b


def c2c(x):
    return torch.from_numpy(np.stack([np.float32(x.real), np.float32(x.imag)], axis=-1))


def r2c(x):
    # convert real to complex
    return torch.stack([x, torch.zeros_like(x)], -1)


def cdiv(x, y):
    # complex division
    a, b = x[..., 0], x[..., 1]
    c, d = y[..., 0], y[..., 1]
    cd2 = c**2 + d**2
    return torch.stack([(a*c+b*d)/cd2, (b*c-a*d)/cd2], -1)


def crdiv(x, y):
    # complex/real division
    a, b = x[..., 0], x[..., 1]
    a = a.to(device)
    b = b.to(device)
    return torch.stack([a/y, b/y], -1)


def csum(x, y):
    # complex + real
    x = x.to(device)
    y = y.to(device)
    return torch.stack([x[..., 0] + y, x[..., 1]], -1)


def cabs(x):
    # modulus of a complex number
    return torch.pow(x[..., 0]**2+x[..., 1]**2, 0.5)


def cabs2(x):
    return x[..., 0]**2+x[..., 1]**2


def cmul(t1, t2):
    '''complex multiplication

    Args:
        t1: NxCxHxWx2, complex tensor
        t2: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    # print('t1',t1.size())
    # print('t2',t2.size())
    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    real1 = real1.to(device)
    real2 = real2.to(device)
    imag1 = imag1.to(device)
    imag2 = imag2.to(device)
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1)


def cconj(t, inplace=False):
    '''complex's conjugation

    Args:
        t: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    c = t.clone() if not inplace else t
    c[..., 1] *= -1
    return c


def rfft(t):
    # Real-to-complex Discrete Fourier Transform
    return torch.fft.fftn(t, dim=[2, 3])


def irfft(t):
    # Complex-to-real Inverse Discrete Fourier Transform
    t = torch.complex(t[..., 0], t[..., 1])
    return torch.fft.ifftn(t, dim=[2, 3])


def fft(t):
    # Complex-to-complex Discrete Fourier Transform
    return torch.fft.fftn(t, dim=[2, 3])


def ifft(t):
    # Complex-to-complex Inverse Discrete Fourier Transform
    t = torch.complex(t[..., 0], t[..., 1])
    return torch.fft.ifftn(t, dim=[2, 3])


def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.

    Args:
        psf: NxCxhxw
        shape: [H, W]

    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.fft.fftn(otf, dim=[2, 3])
    n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    otf = torch.stack((otf.real, otf.imag), dim=-1)
    return otf


def upsample(x, sf=3):
    '''s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf=3):
    '''s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]


def downsample_np(x, sf=3):
    st = 0
    return x[st::sf, st::sf, ...]


"""
# --------------------------------------------
# (1) Prior module; ResUNet: act as a non-blind denoiser
# x_k = P(z_k, beta_k)
# --------------------------------------------
"""


class ResUNet(nn.Module):
    def __init__(self, in_nc=2, out_nc=1, nc=[16, 32, 64, 128], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(ResUNet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):
        
        x = x.to(torch.float32)
        h, w = x.size()[-2:]
        # paddingBottom = int(np.ceil(h/8)*8-h)
        # paddingRight = int(np.ceil(w/8)*8-w)
        # x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        # print(x.size())
        x = self.m_up3(x+x4)
        # print(x.size())
        x = self.m_up2(x+x3)
        # print(x.size())
        x = self.m_up1(x+x2)
        x = torch.nn.functional.interpolate(x, size=(33, 33), mode='bilinear', align_corners=False)
        x = self.m_tail(x+x1)

        x = x[..., :h, :w]

        return x


"""
# --------------------------------------------
# (2) Data module, closed-form solution
# It is a trainable-parameter-free module  ^_^
# z_k = D(x_{k-1}, s, k, y, alpha_k)
# some can be pre-calculated
# --------------------------------------------
"""


class DataNet(nn.Module):
    def __init__(self):
        super(DataNet, self).__init__()

    def forward(self, x, FB, FBC, F2B, FBFy, alpha, sf):
        M = torch.fft.fftn(alpha*x, dim=[2, 3])
        N = torch.stack((M.real, M.imag), dim=-1)
        FR = FBFy + N
        x1 = cmul(FB, FR)
        FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
        invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
        invWBR = cdiv(FBR, csum(invW, alpha))
        FCBinvWBR = cmul(FBC, invWBR.repeat(1, 1, sf, sf, 1))
        FX = (FR-FCBinvWBR)/alpha.unsqueeze(-1)
        FX = torch.complex(FX[..., 0], FX[..., 1])
        Xest = torch.fft.ifftn(FX, dim=[2, 3])

        return Xest


"""
# --------------------------------------------
# (3) Hyper-parameter module
# --------------------------------------------
"""


class HyPaNet(nn.Module):
    def __init__(self, in_nc=2, out_nc=8, channel=64):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
                nn.Conv2d(in_nc, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus())

    def forward(self, x):
        x = self.mlp(x) + 1e-6
        return x


"""
# --------------------------------------------
# main USRNet
# deep unfolding super-resolution network
# --------------------------------------------
"""



class USRNet_bb(nn.Module):
    def __init__(self, n_iter=8, h_nc=64, in_nc=2, out_nc=1, nc=[16, 32, 64, 128], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(USRNet_bb, self).__init__()

        self.d = DataNet()
        self.p = ResUNet(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb, act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode)
        self.h = HyPaNet(in_nc=2, out_nc=n_iter*2, channel=h_nc)
        self.n = n_iter

    def forward(self, x, k, sf, sigma):
        '''
        x: tensor, NxCxWxH
        k: tensor, Nx(1,3)xwxh
        sf: integer, 1
        sigma: tensor, Nx1x1x1
        '''
        # print('k:', k.size())
        # print('sf:', sf)
        # print('sigma:', sigma.size())
        # initialization & pre-calculation
        w, h = x.shape[-2:]
        # print('w', w)
        # print('h', h)
        FB = p2o(k, (w*sf, h*sf))
        # print('FB', FB.size())
        FBC = cconj(FB, inplace=False)
        # print('FBC', FBC.size())
        F2B = r2c(cabs2(FB))
        # print('F2B', F2B.size())
        STy = upsample(x, sf=sf)
        # print('STy', STy.size())
        t2 = torch.fft.fftn(STy, dim=[2, 3])
        t2 = torch.stack((t2.real, t2.imag), dim=-1)
        # print('t2', t2.size())
        FBFy = cmul(FBC, t2)
        # print('FBFy', FBFy.size())
        x = nn.functional.interpolate(x, scale_factor=sf, mode='nearest')

        # hyper-parameter, alpha & beta
        sigma = torch.full((20, 1, 1, 1), sigma, dtype=sigma.dtype, device=sigma.device)
        ab = self.h(torch.cat((sigma, torch.tensor(sf).type_as(sigma).expand_as(sigma)), dim=1))

        # unfolding
        for i in range(self.n):
            
            x = self.d(x, FB, FBC, F2B, FBFy, ab[:, i:i+1, ...], sf)
            x = self.p(torch.cat((x, ab[:, i+self.n:i+self.n+1, ...].repeat(1, 1, x.size(2), x.size(3))), dim=1))

        return x

# class USRNet_bb(nn.Module):
#     def __init__(self, n_iter=8, h_nc=64, in_nc=1, out_nc=1, nc=[16, 32, 64, 128], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
#         super(USRNet_bb, self).__init__()

#         self.d = DataNet()
#         self.p = ResUNet(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb, act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode)
#         self.h = HyPaNet(in_nc=2, out_nc=n_iter*2, channel=h_nc)
#         self.n = n_iter

#     def forward(self, x, k, sf, sigma):
#         '''
#         x: tensor, NxCxWxH
#         k: tensor, Nx(1,3)xwxh
#         sf: integer, 1
#         sigma: tensor, Nx1x1x1
#         '''

#         # initialization & pre-calculation
#         w, h = x.shape[-2:]
        
#         # Adjusting k size to match FB
#         k_resized = torch.nn.functional.interpolate(k, size=(w*sf, h*sf), mode='bilinear', align_corners=False)
#         FB = p2o(k_resized, (w*sf, h*sf))  # Point spread function to optical transfer function
#         FBC = cconj(FB, inplace=False)
#         F2B = r2c(cabs2(FB))
#         STy = upsample(x, sf=sf)
#         FBFy = cmul(FBC, torch.fft.rfft(STy, 2))
#         x = nn.functional.interpolate(x, scale_factor=sf, mode='nearest')

#         # hyper-parameter, alpha & beta
#         sigma = torch.full((20, 1, 1, 1), sigma, dtype=sigma.dtype, device=sigma.device)
#         ab = self.h(torch.cat((sigma, torch.tensor(sf).type_as(sigma).expand_as(sigma)), dim=1))

#         # unfolding
#         for i in range(self.n):
#             x = self.d(x, FB, FBC, F2B, FBFy, ab[:, i:i+1, ...], sf)
#             x = self.p(torch.cat((x, ab[:, i+self.n:i+self.n+1, ...].repeat(1, 1, x.size(2), x.size(3))), dim=1))

#         return x
