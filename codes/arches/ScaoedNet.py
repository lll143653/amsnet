import torch.nn as nn
import torch
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def pixel_unshuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


class PixelUnShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)


class NFAE_Layer(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(NFAE_Layer, self).__init__()

        self.activaton = nn.Sigmoid()

    def forward(self, x):

        var = (x - x.mean(3, keepdim=True).mean(2, keepdim=True)).pow(2)
        spa_ave_var = var.mean(3, keepdim=True).mean(2, keepdim=True)
        cha_ave_var = var.mean(1, keepdim=True)

        y_spa = (10 * var) / (spa_ave_var + 1e-16)
        y_cha = (10 * var) / (cha_ave_var + 1e-16)

        weight_spa = self.activaton(y_spa)
        weight_cha = self.activaton(y_cha)
        weight = weight_spa * weight_cha

        return x * weight


class DE_Net(nn.Module):
    def __init__(self, nf=64, nf_2=64, input_para=1, num_blocks=5):
        super(DE_Net, self).__init__()
        self.head_noisy = BasicConv(
            1*4, nf_2, 3, stride=1, padding=(3 - 1) // 2, relu=False)
        self.head_Q = BasicConv(64, nf_2, 3, stride=1,
                                padding=(3 - 1) // 2, relu=False)
        self.m_down = PixelUnShuffle(upscale_factor=2)
        self.m_up = nn.PixelShuffle(2)

        self.ConvNet_Input = nn.Sequential(*[
            BasicConv(nf*2, nf, 1, stride=1, padding=(1 - 1) // 2, relu=False),
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True)
        ])
        self.ConvNet_f0noise = nn.Sequential(*[
            BasicConv(nf*2, nf, 1, stride=1, padding=(1 - 1) // 2, relu=False),
        ])
        self.att = NFAE_Layer()
        self.conv1 = BasicConv(nf, input_para*4, 3,
                               stride=1, padding=(3 - 1) // 2, relu=False)

    def forward(self, u, Q, z):
        Q_down = self.head_Q(Q)
        u = self.head_noisy(self.m_down(u))
        cat_Q_z = self.ConvNet_f0noise(torch.cat((Q_down, z), dim=1))
        cat_input = self.ConvNet_Input(torch.cat((u, cat_Q_z), dim=1))
        return self.m_up(self.conv1(self.att(cat_input)))


class ini_DE_Net(nn.Module):
    def __init__(self, in_nc=3, nf=64):
        super(ini_DE_Net, self).__init__()

        self.ConvNet = nn.Sequential(*[
            BasicConv(in_nc*4, nf, 3, stride=1,
                      padding=(3 - 1) // 2, relu=True),
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True),
            BasicConv(nf, nf, 3, stride=1, padding=(3 - 1) // 2, relu=True)
        ])
        self.m_down = PixelUnShuffle(upscale_factor=2)
        self.m_up = nn.PixelShuffle(2)
        self.att = NFAE_Layer()
        self.conv1 = BasicConv(nf, 1*4, 3, stride=1,
                               padding=(3 - 1) // 2, relu=False)

    def forward(self, y):
        y_down = self.m_down(y)
        Q = self.att(self.ConvNet(y_down))
        u0 = self.m_up(self.conv1(Q))
        return u0, Q


class SidePool(nn.Module):
    def forward(self, x, a):
        return torch.cat((torch.mean(x, a).unsqueeze(a), torch.max(x, a)[0].unsqueeze(a)), dim=a)


class ChannelPool_2D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelPool_2D, self).__init__()
        self.Conv = BasicConv(channel, 2, 1, stride=1,
                              padding=(1 - 1) // 2, relu=False)

    def forward(self, x):
        return torch.cat((torch.mean(x, 1).unsqueeze(1),
                          torch.max(x, 1)[0].unsqueeze(1),
                          self.Conv(x)), dim=1)


class ChannelPool_1D(nn.Module):
    def forward(self, x):
        return torch.cat((x.mean(3).mean(2, keepdim=True), x.max(3)[0].max(2, keepdim=True)[0]), 2)


class inplaceCA(nn.Module):
    def __init__(self, channel):
        super(inplaceCA, self).__init__()
        self.Conv = BasicConv(
            4, 4, 1, stride=1, padding=(1 - 1) // 2, relu=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        y = self.Conv(y)
        return x + x*self.sig(y)


class CIG_SA_attention(nn.Module):
    def __init__(self, channel):
        super(CIG_SA_attention, self).__init__()
        self.compress = ChannelPool_2D(channel)
        self.inplaceCA = inplaceCA(channel)
        self.conv_du = nn.Sequential(
            BasicConv(4, 4, 3, stride=1, padding=(
                3 - 1) // 2, relu=False, bias=True),
            self.inplaceCA,
            BasicConv(4, 1, 1, stride=1, padding=(
                1 - 1) // 2, relu=False, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.compress(x)
        y = self.conv_du(y)
        return x * y


class inplaceSA(nn.Module):
    def __init__(self, channel):
        super(inplaceSA, self).__init__()
        self.Conv = BasicConv(channel, 1, 3, stride=1,
                              padding=(3 - 1) // 2, relu=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.Conv(x)
        x = x * self.sig(y)
        return torch.cat((x.mean(3).mean(2, keepdim=True), x.max(3)[0].max(2, keepdim=True)[0]), 2)


class CIG_CA_attention(nn.Module):
    def __init__(self, channel):
        super(CIG_CA_attention, self).__init__()
        self.compress = ChannelPool_1D()
        self.inplaceSA = inplaceSA(channel)
        self.cat = nn.Sequential(
            BasicConv(channel, 4, 1, stride=1,
                      padding=(1 - 1) // 2, relu=True),
            BasicConv(4, channel, 1, stride=1,
                      padding=(1 - 1) // 2, relu=False)
        )
        self.conv_du = nn.Sequential(
            BasicConv(4, 1, 1, stride=1, padding=(1 - 1) // 2, relu=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        a = self.compress(x)
        b = self.inplaceSA(x)

        y = self.cat(torch.cat((a, b), 2).unsqueeze(3))
        y = self.conv_du(y.transpose(1, 2)).transpose(1, 2)
        return x * y


class FMMA_attention(nn.Module):
    """
    FM2ARB attention module

    Args:
        nn (_type_): 这个块会被重复调用16次
    """

    def __init__(self, channel):
        super(FMMA_attention, self).__init__()
        self.head = nn.Sequential(
            BasicConv(channel, channel, 3, stride=1,
                      padding=(3 - 1) // 2, relu=True),
            BasicConv(channel, channel, 3, stride=1,
                      padding=(3 - 1) // 2, relu=True),
            BasicConv(channel, channel, 3, stride=1,
                      padding=(3 - 1) // 2, relu=True),
            BasicConv(channel, channel, 3, stride=1,
                      padding=(3 - 1) // 2, relu=False)
        )
        self.DEM = nn.Sequential(
            BasicConv(channel, channel, 1, stride=1,
                      padding=(1 - 1) // 2, relu=True),
            BasicConv(channel, channel, 1, stride=1,
                      padding=(1 - 1) // 2, relu=False),
            nn.Sigmoid()
        )

        self.CIG_CA = CIG_CA_attention(channel)  # Spatial Attention
        self.CIG_SA = CIG_SA_attention(channel)  # Channel Attention

        self.cat = BasicConv(channel*2, channel, 1, stride=1,
                             padding=(1 - 1) // 2, relu=False)

    def forward(self, y):
        a, g = y
        up_a = self.head(a)
        ca_branch = self.CIG_CA(up_a)
        sa_branch = self.CIG_SA(up_a)
        mix = self.cat(torch.cat((ca_branch, sa_branch), 1))
        b = torch.mul(mix, self.DEM(g)) + a

        return b, g


class RE_Net(nn.Module):
    def __init__(self):
        super(RE_Net, self).__init__()
        num_crb = 16
        para = 1

        n_feats = 64
        kernel_size = 3
        inp_chans = 1  # 4 RGGB channels, and 4 Variance maps

        modules_head = [
            BasicConv(n_feats*2, n_feats, 1, stride=1,
                      padding=(1 - 1) // 2, relu=False)
        ]

        modules_body = [
            FMMA_attention(n_feats)
            for i in range(num_crb)]

        modules_tail = [
            BasicConv(n_feats, n_feats, kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, relu=False)
        ]

        self.head_E = BasicConv(n_feats, n_feats, 3,
                                stride=1, padding=(3 - 1) // 2, relu=False)

        self.head_noisy = BasicConv(
            inp_chans*4, n_feats, 3, stride=1, padding=(3 - 1) // 2, relu=False)
        self.head_z = BasicConv(n_feats, n_feats, 3,
                                stride=1, padding=(3 - 1) // 2, relu=False)
        self.GR = nn.Sequential(*[
            BasicConv(n_feats, n_feats, 3, stride=1,
                      padding=(3 - 1) // 2, relu=True),
            BasicConv(n_feats, n_feats, 3, stride=1,
                      padding=(3 - 1) // 2, relu=True),
            BasicConv(n_feats, n_feats, 3, stride=1,
                      padding=(3 - 1) // 2, relu=True),
            BasicConv(n_feats, n_feats, 3, stride=1,
                      padding=(3 - 1) // 2, relu=False),
        ])

        self.m_down = PixelUnShuffle(upscale_factor=2)
        self.m_up = nn.PixelShuffle(2)
        self.fusion = nn.Sequential(*modules_head)
        self.y_u_cat = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.Conv = BasicConv(n_feats, n_feats, 3, stride=1,
                              padding=(3 - 1) // 2, relu=False)
        self.DEM = nn.Sequential(
            BasicConv(n_feats, n_feats, 3, stride=1,
                      padding=(3 - 1) // 2, relu=True),
            BasicConv(n_feats, n_feats, 3, stride=1,
                      padding=(3 - 1) // 2, relu=False),
            nn.Sigmoid()
        )
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, z, E, u):
        z = self.head_z(z)
        E = self.head_E(E)
        u = self.head_noisy(self.m_down(u))
        cat_y_u = self.y_u_cat(torch.cat((E, u), dim=1))
        fusion = self.fusion(torch.cat((z, cat_y_u), dim=1))
        g = self.GR(u)
        inputs = [fusion, g]
        b, _ = self.body(inputs)
        b = torch.mul(self.Conv(b), self.DEM(g))
        return self.tail(b+z)


class SCNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, para=1, T=3):
        super(SCNet, self).__init__()
        self.head = nn.Sequential(
            BasicConv(in_nc*4, nf, 3, stride=1,
                      padding=(3 - 1) // 2, relu=False)
        )

        self.m_down = PixelUnShuffle(upscale_factor=2)
        self.m_up = nn.PixelShuffle(2)

        self.T = T
        self.DE_Net = DE_Net()
        self.ini_DE_Net = ini_DE_Net(in_nc=3, nf=nf)
        self.RE_Net = RE_Net()  # G-Module is implemented in RE_Net

        self.tail = nn.Sequential(
            BasicConv(nf, out_nc*4, 3, stride=1,
                      padding=(3 - 1) // 2, relu=False)
        )

        self.a = nn.Parameter(torch.ones(1) * 1)
        self.b = nn.Parameter(torch.ones(1) * 1)

    def forward(self, noisyImage):
        E = self.head(self.m_down(noisyImage))
        u, Q = self.ini_DE_Net(noisyImage)
        z = self.a*self.RE_Net(E, E, u) + self.b*E
        outs = []
        outs.append(u)

        for i in range(self.T):
            u = u + self.DE_Net(u, Q, z)
            z = self.a*self.RE_Net(z, E, u) + self.b*E
            outs.append(u)
        return self.m_up(self.tail(z))

    def denoise(self, noisyImage):
        return self.forward(noisyImage)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # torch.nn.init.xavier_normal_(m.weight.data)
                torch.nn.init.xavier_uniform_(m.weight.data)
                # torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
