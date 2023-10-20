import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50, resnet18
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from CST import  CST_module

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio//reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)



        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)



class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.l = nn.Conv2d(channel, channel, 1)

    def forward(self, x, f):
        y = self.fc(x)
        return f * y + f



class SENet_learning(nn.Module):
    def __init__(self, dim):
        super(SENet_learning, self).__init__()

        self.se_1 = SELayer(dim)
        self.se_2 = SELayer(dim)
        self.se_3 = SELayer(dim)
        self.se_4 = SELayer(dim)
        self.se_5 = SELayer(dim)
        self.se_6 = SELayer(dim)

        self.se_7 = SELayer(dim)
        self.se_8 = SELayer(dim)
        self.se_9 = SELayer(dim)
        self.se_10 = SELayer(dim)
        self.se_11 = SELayer(dim)
        self.se_12 = SELayer(dim)
        self.se_13 = SELayer(dim)
        self.se_14 = SELayer(dim)



        self.a = nn.Linear(dim, dim)
        self.b = nn.Linear(dim, dim)
        self.c = nn.Linear(dim, dim)
        self.d = nn.Linear(dim, dim)
        self.e = nn.Linear(dim, dim)
        self.f = nn.Linear(dim, dim)
        self.g = nn.Linear(dim, dim)
        self.h = nn.Linear(dim, dim)
        self.i = nn.Linear(dim, dim)
        self.j = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.l = nn.Linear(dim, dim)
        self.m = nn.Linear(dim, dim)
        self.n = nn.Linear(dim, dim)

    def forward(self, t_x, x, x_h):
        local_feat_list = []

        t1 = (self.a(t_x) + x[0]) / 2
        t2 = (self.b(t_x) + x[1]) / 2
        t3 = (self.c(t_x) + x[2]) / 2
        t4 = (self.d(t_x) + x[3]) / 2
        t5 = (self.e(t_x) + x[4]) / 2
        t6 = (self.f(t_x) + x[5]) / 2
        t7 = (self.g(t_x) + x[6]) / 2
        t8 = (self.h(t_x) + x[7]) / 2
        t9 = (self.i(t_x) + x[8]) / 2
        t10 = (self.j(t_x) + x[9]) / 2

        t11 = (self.k(t_x) + x[10]) / 2
        t12 = (self.l(t_x) + x[11]) / 2
        t13 = (self.m(t_x) + x[12]) / 2
        t14 = (self.n(t_x) + x[13]) / 2

        f1 = self.se_1(t1, x_h[0])
        f2 = self.se_2(t2, x_h[1])
        f3 = self.se_3(t3, x_h[2])
        f4 = self.se_4(t4, x_h[3])
        f5 = self.se_5(t5, x_h[4])
        f6 = self.se_6(t6, x_h[5])
        f7 = self.se_7(t7, x_h[6])
        f8 = self.se_8(t8, x_h[7])
        f9 = self.se_9(t9, x_h[8])
        f10 = self.se_10(t10, x_h[9])

        f11 = self.se_11(t11, x_h[10])
        f12 = self.se_12(t12, x_h[11])
        f13 = self.se_13(t13, x_h[12])
        f14 = self.se_14(t14, x_h[13])


        f1 = f1.unsqueeze(dim=1)  ###[32, 1, [2048]]
        f2 = f2.unsqueeze(dim=1)
        f3 = f3.unsqueeze(dim=1)
        f4 = f4.unsqueeze(dim=1)
        f5 = f5.unsqueeze(dim=1)
        f6 = f6.unsqueeze(dim=1)
        f7 = f7.unsqueeze(dim=1)
        f8 = f8.unsqueeze(dim=1)
        f9 = f9.unsqueeze(dim=1)
        f10 = f10.unsqueeze(dim=1)

        f11 = f11.unsqueeze(dim=1)
        f12 = f12.unsqueeze(dim=1)
        f13 = f13.unsqueeze(dim=1)
        f14 = f14.unsqueeze(dim=1)

        f = torch.cat((f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14), dim=1)
        f = f.mean(dim=1)

        return f


class embed_net(nn.Module):
    def __init__(self, low_dim, class_num, share_net, h_size, w_size, patch_size_h, patch_size_w, frame, frame_size,
                 pcb, local_feat_dim=256, drop=0.2, part=3, alpha=0.2, nheads=4,
                 arch='resnet50', wpa=False):
        super(embed_net, self).__init__()

        self.non_local = 'on'
        self.gm_pool = 'on'


        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)



        # self.non_local = no_local
        if self.non_local =='on':
            layers=[3, 4, 6, 3]
            non_layers=[0,2,3,0]
            self.NL_1 = nn.ModuleList(
                [Non_local(256) for i in range(non_layers[0])])
            self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
            self.NL_2 = nn.ModuleList(
                [Non_local(512) for i in range(non_layers[1])])
            self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
            self.NL_3 = nn.ModuleList(
                [Non_local(1024) for i in range(non_layers[2])])
            self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
            self.NL_4 = nn.ModuleList(
                [Non_local(2048) for i in range(non_layers[3])])
            self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])


        pool_dim = 2048
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.SENet_learning = SENet_learning(local_feat_dim)
        # self.gm_pool = gm_pool





        self.cst_module = CST_module(
            image_size_h=h_size,
            image_size_w=w_size,
            patch_size_h=patch_size_h,
            patch_size_w=patch_size_w,
            frames=frame,
            frame_patch_size=frame_size,  # frame patch size
            num_classes=class_num,
            dim=local_feat_dim,
            depth=1,
            heads=128,
            mlp_dim=2048,
            dim_head=48,
            channels=2048,
            dropout=0.1,
            emb_dropout=0.1
        )



    def forward(self, x1, x2, modal, seq_len):

        if seq_len > 1:
            b, c, h, w = x1.size()
            t = seq_len

            x1 = x1.view(int(b * seq_len), int(c / seq_len), h, w)
            x2 = x2.view(int(x2.size()[0] * seq_len), int(c / seq_len), h, w)

        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        # shared block
        if self.non_local == 'on':
            NL1_counter = 0
            if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
            for i in range(len(self.base_resnet.base.layer1)):
                x = self.base_resnet.base.layer1[i](x)
                if i == self.NL_1_idx[NL1_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_1[NL1_counter](x)
                    NL1_counter += 1
            # Layer 2
            NL2_counter = 0
            if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
            for i in range(len(self.base_resnet.base.layer2)):
                x = self.base_resnet.base.layer2[i](x)
                if i == self.NL_2_idx[NL2_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_2[NL2_counter](x)
                    NL2_counter += 1
            # Layer 3
            NL3_counter = 0
            if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
            for i in range(len(self.base_resnet.base.layer3)):
                x = self.base_resnet.base.layer3[i](x)
                if i == self.NL_3_idx[NL3_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_3[NL3_counter](x)
                    NL3_counter += 1
            # Layer 4
            NL4_counter = 0
            if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
            for i in range(len(self.base_resnet.base.layer4)):
                x = self.base_resnet.base.layer4[i](x)
                if i == self.NL_4_idx[NL4_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_4[NL4_counter](x)
                    NL4_counter += 1
        else:
            x = self.base_resnet(x)
        if self.gm_pool  == 'on':
            b, c, h, w = x.shape
            x_fu = x
            x_pcb = x
            x_pure = x
            x = x.view(b, c, -1)
            p = 3.0
            x_pool = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p)
        else:
            x_pool = self.avgpool(x)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))


        x_fu = x_fu.view(int(x_fu.size(0) // t), t, x_fu.shape[1], x_fu.shape[2], x_fu.shape[3]).permute(0, 2, 1, 3, 4) ###

        x_fusion,_ = self.cst_module(x_fu)

        x_pool_frame = x_pool.view(int(x_pool.size(0)/ t), t, -1).permute(1, 0, 2)
        x_pool_frame_fuse = self.SENet_learning(x_fusion, x_pool_frame, x_pool_frame)
        x_pool = x_pool_frame_fuse

        feat = self.bottleneck(x_pool)

        if self.training:
            return x_pool, self.classifier(feat)
        else:
            return  self.l2norm(feat)
