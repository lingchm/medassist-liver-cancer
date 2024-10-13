
import torch
import torch.nn as nn
import torch.nn.functional as F


# 2D: net = UNet2D(1,2,pab_channels=64,use_batchnorm=True)
# 3D: net = UNet3D(1,2,pab_channels=32,use_batchnorm=True)

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

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
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)



class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):

        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()

        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):

        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()

        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm3d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv3dReLU, self).__init__(conv, bn, relu)
class PAB2D(nn.Module):
    def __init__(self, in_channels, out_channels, pab_channels=64):
        super(PAB2D, self).__init__()
        # Series of 1x1 conv to generate attention feature maps
        self.pab_channels = pab_channels
        self.in_channels = in_channels
        self.top_conv = nn.Conv2d(in_channels, pab_channels, kernel_size=1)
        self.center_conv = nn.Conv2d(in_channels, pab_channels, kernel_size=1)
        self.bottom_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.map_softmax = nn.Softmax(dim=1)
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        bsize = x.size()[0]
        h = x.size()[2]
        w = x.size()[3]
        x_top = self.top_conv(x)
        x_center = self.center_conv(x)
        x_bottom = self.bottom_conv(x)

        x_top = x_top.flatten(2)
        x_center = x_center.flatten(2).transpose(1, 2)
        x_bottom = x_bottom.flatten(2).transpose(1, 2)

        sp_map = torch.matmul(x_center, x_top)
        sp_map = self.map_softmax(sp_map.view(bsize, -1)).view(bsize, h*w, h*w)
        sp_map = torch.matmul(sp_map, x_bottom)
        sp_map = sp_map.reshape(bsize, self.in_channels, h, w)
        x = x + sp_map
        x = self.out_conv(x)
        # print('x_top',x_top.shape,'x_center',x_center.shape,'x_bottom',x_bottom.shape,'x',x.shape,'sp_map',sp_map.shape)
        return x

class MFAB2D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True, reduction=16):
        # MFAB is just a modified version of SE-blocks, one for skip, one for input
        super(MFAB2D, self).__init__()
        self.hl_conv = nn.Sequential(
            Conv2dReLU(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                use_batchnorm=use_batchnorm,
            ),
            Conv2dReLU(
                in_channels,
                skip_channels,
                kernel_size=1,
                use_batchnorm=use_batchnorm,
            )
        )
        self.SE_ll = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(skip_channels, skip_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(skip_channels // reduction, skip_channels, 1),
            nn.Sigmoid(),
        )
        self.SE_hl = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(skip_channels, skip_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(skip_channels // reduction, skip_channels, 1),
            nn.Sigmoid(),
        )
        self.conv1 = Conv2dReLU(
            skip_channels + skip_channels,  # we transform C-prime form high level to C from skip connection
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x, skip=None):
        x = self.hl_conv(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        attention_hl = self.SE_hl(x)
        if skip is not None:
            attention_ll = self.SE_ll(skip)
            attention_hl = attention_hl + attention_ll
            x = x * attention_hl
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class PAB3D(nn.Module):
    def __init__(self, in_channels, out_channels, pab_channels=64):
        super(PAB3D, self).__init__()
        # Series of 1x1 conv to generate attention feature maps
        self.pab_channels = pab_channels
        self.in_channels = in_channels
        self.top_conv = nn.Conv3d(in_channels, pab_channels, kernel_size=1)
        self.center_conv = nn.Conv3d(in_channels, pab_channels, kernel_size=1)
        self.bottom_conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.map_softmax = nn.Softmax(dim=1)
        self.out_conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        bsize = x.size()[0]
        h = x.size()[2]
        w = x.size()[3]
        d = x.size()[4]
        x_top = self.top_conv(x)
        x_center = self.center_conv(x)
        x_bottom = self.bottom_conv(x)

        x_top = x_top.flatten(2)
        x_center = x_center.flatten(2).transpose(1, 2)
        x_bottom = x_bottom.flatten(2).transpose(1, 2)
        sp_map = torch.matmul(x_center, x_top)
        sp_map = self.map_softmax(sp_map.view(bsize, -1)).view(bsize, h*w*d, h*w*d)
        sp_map = torch.matmul(sp_map, x_bottom)
        sp_map = sp_map.reshape(bsize, self.in_channels, h, w, d)
        x = x + sp_map
        x = self.out_conv(x)
        # print('x_top',x_top.shape,'x_center',x_center.shape,'x_bottom',x_bottom.shape,'x',x.shape,'sp_map',sp_map.shape)
        return x

class MFAB3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True, reduction=16):
        # MFAB is just a modified version of SE-blocks, one for skip, one for input
        super(MFAB3D, self).__init__()
        self.hl_conv = nn.Sequential(
            Conv3dReLU(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                use_batchnorm=use_batchnorm,
            ),
            Conv3dReLU(
                in_channels,
                skip_channels,
                kernel_size=1,
                use_batchnorm=use_batchnorm,
            )
        )
        self.SE_ll = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(skip_channels, skip_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(skip_channels // reduction, skip_channels, 1),
            nn.Sigmoid(),
        )
        self.SE_hl = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(skip_channels, skip_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(skip_channels // reduction, skip_channels, 1),
            nn.Sigmoid(),
        )
        self.conv1 = Conv3dReLU(
            skip_channels + skip_channels,  # we transform C-prime form high level to C from skip connection
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x, skip=None):
        x = self.hl_conv(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        attention_hl = self.SE_hl(x)
        if skip is not None:
            attention_ll = self.SE_ll(skip)
            attention_hl = attention_hl + attention_ll
            x = x * attention_hl
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DoubleConv2D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down2D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            NONLocalBlock2D(in_channels),
            DoubleConv2D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up2D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv2D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv2D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet2D(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, pab_channels=64, use_batchnorm=True, aux_classifier = False):
        super(UNet2D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv2D(n_channels, pab_channels)
        self.down1 = Down2D(pab_channels, 2*pab_channels)
        self.down2 = Down2D(2*pab_channels, 4*pab_channels)
        self.down3 = Down2D(4*pab_channels, 8*pab_channels)
        factor = 2 if bilinear else 1
        self.down4 = Down2D(8*pab_channels, 16*pab_channels // factor)
        self.pab = PAB2D(8*pab_channels,8*pab_channels)
        self.up1 = Up2D(16*pab_channels, 8*pab_channels // factor, bilinear)
        self.up2 = Up2D(8*pab_channels, 4*pab_channels // factor, bilinear)
        self.up3 = Up2D(4*pab_channels, 2*pab_channels // factor, bilinear)
        self.up4 = Up2D(2*pab_channels, pab_channels, bilinear)

        self.mfab1 = MFAB2D(8*pab_channels,8*pab_channels,4*pab_channels,use_batchnorm)
        self.mfab2 = MFAB2D(4*pab_channels,4*pab_channels,2*pab_channels,use_batchnorm)
        self.mfab3 = MFAB2D(2*pab_channels,2*pab_channels,pab_channels,use_batchnorm)
        self.mfab4 = MFAB2D(pab_channels,pab_channels,pab_channels,use_batchnorm)
        self.outc = OutConv2D(pab_channels, n_classes)
        
        if aux_classifier == False:
          self.aux = None
        else:
          # customize the auxiliary classification loss
          # self.aux = nn.Sequential(nn.AdaptiveAvgPool2d(1),
          #                          nn.Flatten(),
          #                          nn.Dropout(p=0.1, inplace=True),
          #                          nn.Linear(8*pab_channels, 16, bias=True),
          #                          nn.Dropout(p=0.1, inplace=True),
          #                          nn.Linear(16, n_classes, bias=True),
          #                          nn.Softmax(1))

          self.aux = nn.Sequential(
                                   NONLocalBlock2D(8*pab_channels),
                                   nn.Conv2d(8*pab_channels,1,1),
                                   nn.InstanceNorm2d(1),
                                   nn.ReLU(),       
                                   nn.Flatten(),
                                   nn.Linear(24*24, 16, bias=True),
                                   nn.Dropout(p=0.2, inplace=True),
                                   nn.Linear(16, n_classes, bias=True),
                                   nn.Softmax(1))
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.pab(x5)
        
        x = self.mfab1(x5,x4)
        x = self.mfab2(x,x3)
        x = self.mfab3(x,x2)
        x = self.mfab4(x,x1)

        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        logits = self.outc(x)
        logits = F.softmax(logits,1)

        if self.aux ==None:
          return logits
        else:
          aux = self.aux(x5)
          return logits, aux
          
          
          
          
class DoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            # NONLocalBlock3D(in_channels),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, pab_channels=64, use_batchnorm=True, aux_classifier = False):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv3D(n_channels, pab_channels)
        self.down1 = Down3D(pab_channels, 2*pab_channels)
        self.nnblock2 = NONLocalBlock3D(2*pab_channels)
        self.down2 = Down3D(2*pab_channels, 4*pab_channels)
        self.down3 = Down3D(4*pab_channels, 8*pab_channels)
        factor = 2 if bilinear else 1
        self.down4 = Down3D(8*pab_channels, 16*pab_channels // factor)
        self.pab = PAB3D(8*pab_channels,8*pab_channels)
        self.up1 = Up3D(16*pab_channels, 8*pab_channels // factor, bilinear)
        self.up2 = Up3D(8*pab_channels, 4*pab_channels // factor, bilinear)
        self.up3 = Up3D(4*pab_channels, 2*pab_channels // factor, bilinear)
        self.up4 = Up3D(2*pab_channels, pab_channels, bilinear)

        self.mfab1 = MFAB3D(8*pab_channels,8*pab_channels,4*pab_channels,use_batchnorm)
        self.mfab2 = MFAB3D(4*pab_channels,4*pab_channels,2*pab_channels,use_batchnorm)
        self.mfab3 = MFAB3D(2*pab_channels,2*pab_channels,pab_channels,use_batchnorm)
        self.mfab4 = MFAB3D(pab_channels,pab_channels,pab_channels,use_batchnorm)
        self.outc = OutConv3D(pab_channels, n_classes)

        if aux_classifier == False:
          self.aux = None
        else:
          # customize the auxiliary classification loss
          # self.aux = nn.Sequential(nn.AdaptiveMaxPool3d(1),
          #                          nn.Flatten(),
          #                          nn.Dropout(p=0.1, inplace=True),
          #                          nn.Linear(8*pab_channels, 16, bias=True),
          #                          nn.Dropout(p=0.1, inplace=True),
          #                          nn.Linear(16, n_classes, bias=True),
          #                          nn.Softmax(1))
          
          self.aux = nn.Sequential(nn.Conv3d(8*pab_channels,1,1),   
                                   nn.InstanceNorm3d(1),
                                   nn.ReLU(),       
                                   nn.Flatten(),
                                   nn.Linear(16*16*2, 16, bias=True),
                                   nn.Dropout(p=0.2, inplace=True),
                                   nn.Linear(16, n_classes, bias=True),
                                   nn.Softmax(1))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        # x2 = self.nnblock2(x2)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.pab(x5)
        
        x = self.mfab1(x5,x4)
        x = self.mfab2(x,x3)
        x = self.mfab3(x,x2)
        x = self.mfab4(x,x1)

        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        logits = self.outc(x)
        logits = F.softmax(logits,1)

        if self.aux ==None:
          return logits
        else:
          aux = self.aux(x5)
          return logits, aux