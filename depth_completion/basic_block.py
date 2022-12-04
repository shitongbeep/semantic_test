import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from torchvision.transforms import transforms


def weights_init(m):
    #  Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False, padding=1):
    """3x3 convolution with padding"""
    if padding >= 1:
        padding = dilation
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, groups=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=bias)


def convbnrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                         nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


def deconvbnrelu(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels,
                           out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           output_padding=output_padding,
                           bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


def convbn(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                         nn.BatchNorm2d(out_channels))


def deconvbn(in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels,
                           out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           output_padding=output_padding,
                           bias=False), nn.BatchNorm2d(out_channels))


class GeometryFeature(nn.Module):
    '''从归一化uv坐标还原相机坐标系下xyz坐标'''

    def __init__(self):
        super(GeometryFeature, self).__init__()

    def forward(self, z, vnorm, unorm, h, w, ch, cw, fh, fw):
        x = z * (0.5 * h * (vnorm + 1) - ch) / fh
        y = z * (0.5 * w * (unorm + 1) - cw) / fw
        return torch.cat((x, y, z), 1)


class SparseDownSampleClose(nn.Module):

    def __init__(self, stride):
        super(SparseDownSampleClose, self).__init__()
        self.pooling = nn.MaxPool2d(stride, stride)
        self.large_number = 600

    def forward(self, d, mask):
        encode_d = -(1 - mask) * self.large_number - d

        d = -self.pooling(encode_d)
        mask_result = self.pooling(mask)
        d_result = d - (1 - mask_result) * self.large_number

        return d_result, mask_result


class SparseDownSample(nn.Module):

    def __init__(self, kernel_size, stride):
        super(SparseDownSample, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.depth_pad = transforms.Pad([0, 0, 1, 1], 0)
        self.geo_pad = transforms.Pad([0, 0, 1, 1], 0)

    def forward(self, depth, geo_s):
        B = depth.shape[0]
        C = geo_s.shape[1]
        H = depth.shape[-2]
        W = depth.shape[-1]
        if H % 2 or W % 2:
            depth = self.depth_pad(depth)
            geo_s = self.geo_pad(geo_s)

        depth_unfold = F.unfold(depth, kernel_size=self.kernel_size, stride=self.stride)
        geo_unfold = F.unfold(geo_s, kernel_size=self.kernel_size, stride=self.stride)
        _, index = depth_unfold.topk(1, dim=1)
        all_index = []
        for i in range(C):
            all_index.append(index + i * self.kernel_size**2)
        all_index = torch.cat(all_index, dim=1)
        depth_fold = torch.gather(depth_unfold, dim=1, index=index).view(B, 1, (H + 1) // 2, (W + 1) // 2)
        geo_unfold = torch.gather(geo_unfold, dim=1, index=all_index).view(B, C, (H + 1) // 2, (W + 1) // 2)

        return depth_fold, geo_unfold


class gtDownSample(pl.LightningModule):

    def __init__(self, stride):
        super(gtDownSample, self).__init__()
        self.pooling = nn.MaxPool2d(kernel_size=stride, stride=stride)

    def forward(self, gt_mask):
        mask_result = self.pooling(gt_mask)
        return mask_result


class BasicBlockGeo(pl.LightningModule):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, geoplanes=3):
        super(BasicBlockGeo, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            # norm_layer = encoding.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        #  Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes + geoplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes + geoplanes, planes)
        self.bn2 = norm_layer(planes)
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes + geoplanes, planes, stride),
                norm_layer(planes),
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, g1=None, g2=None):
        identity = x
        if g1 is not None:
            x = torch.cat((x, g1), 1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if g2 is not None:
            out = torch.cat((g2, out), 1)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DepthLeanerBlock(pl.LightningModule):

    def __init__(self, studentPlanes, teacherPlanes):
        super(DepthLeanerBlock, self).__init__()
        if studentPlanes != teacherPlanes:
            raise RuntimeError("in basic_block.py : when fusion, teacher and student network features size are not equal")
        self.hiden_size = studentPlanes

        self.learners = nn.Conv2d(self.hiden_size, self.hiden_size, kernel_size=3, padding=1)
        self.fcs1 = nn.Conv2d(self.hiden_size * 2, self.hiden_size, kernel_size=3, padding=1)
        self.fcs2 = nn.Conv2d(self.hiden_size, self.hiden_size, kernel_size=3, padding=1)

    def forward(self, studentFeature, teacherFeature):
        feat_learner = self.learners(studentFeature)
        feat_learner = F.relu(feat_learner)

        feat_cat = torch.cat([teacherFeature, feat_learner], dim=1)
        feat_cat = self.fcs1(feat_cat)
        feat_weight = torch.sigmoid(self.fcs2(feat_cat))
        feat_cat = F.relu(feat_cat * feat_weight)
        teacher_feat_fusion = feat_cat + teacherFeature

        return teacher_feat_fusion


# * CSPN 还没看懂
def makePad(gks):
    pad = []
    for i in range(gks):
        for j in range(gks):
            top = i
            bottom = gks - 1 - i
            left = j
            right = gks - 1 - j
            pad.append(torch.nn.ZeroPad2d((left, right, top, bottom)))
    return pad


class CSPNGenerate(nn.Module):

    def __init__(self, in_channels, kernel_size):
        super(CSPNGenerate, self).__init__()
        self.kernel_size = kernel_size
        self.generate = convbn(in_channels, self.kernel_size * self.kernel_size - 1, kernel_size=3, stride=1, padding=1)

    def forward(self, feature):

        guide = self.generate(feature)  # 卷积核-1个权重层

        # normalization
        guide_sum = torch.sum(guide.abs(), dim=1).unsqueeze(1)
        guide = torch.div(guide, guide_sum)  # 所有权重层绝对值归一化
        guide_mid = (1 - torch.sum(guide, dim=1)).unsqueeze(1)  # 中间位置权重层补齐1

        # padding
        zero_pad = makePad(self.kernel_size)
        weight_pad = []
        for t in range(self.kernel_size * self.kernel_size):
            if (t < int((self.kernel_size * self.kernel_size - 1) / 2)):
                weight_pad.append(zero_pad[t](guide[:, t:t + 1, :, :]))
            elif (t > int((self.kernel_size * self.kernel_size - 1) / 2)):
                weight_pad.append(zero_pad[t](guide[:, t - 1:t, :, :]))
            else:
                weight_pad.append(zero_pad[t](guide_mid))

        guide_weight = torch.cat([weight_pad[t] for t in range(self.kernel_size * self.kernel_size)], dim=1)
        return guide_weight


class CSPN(nn.Module):

    def __init__(self, kernel_size):
        super(CSPN, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, guide_weight, hn, h0):

        # CSPN
        half = (self.kernel_size * self.kernel_size - 1) // 2
        zero_pad = makePad(self.kernel_size)
        result_pad = []
        for t in range(self.kernel_size * self.kernel_size):
            if (t == half):
                result_pad.append(zero_pad[t](h0))  # kernel size 的中心
            else:
                result_pad.append(zero_pad[t](hn))  # kernel size 的其他位置
        guide_result = torch.cat([result_pad[t] for t in range(self.kernel_size * self.kernel_size)], dim=1)

        guide_result = torch.sum((guide_weight.mul(guide_result)), dim=1)
        guide_result = guide_result[:, (self.kernel_size - 1) // 2:-(self.kernel_size - 1) // 2,
                                    (self.kernel_size - 1) // 2:-(self.kernel_size - 1) // 2]  # 只要kernel size里面的，pad的东西不要

        return guide_result.unsqueeze(dim=1)


class CSPNGenerateAccelerate(nn.Module):

    def __init__(self, in_channels, kernel_size):
        super(CSPNGenerateAccelerate, self).__init__()
        self.kernel_size = kernel_size
        self.generate = convbn(in_channels, self.kernel_size * self.kernel_size - 1, kernel_size=3, stride=1, padding=1)

    def forward(self, feature):

        guide = self.generate(feature)

        # normalization in standard CSPN
        # '''
        guide_sum = torch.sum(guide.abs(), dim=1).unsqueeze(1)
        guide = torch.div(guide, guide_sum)
        guide_mid = (1 - torch.sum(guide, dim=1)).unsqueeze(1)
        # '''
        # weight_pad = [i for i in range(self.kernel_size * self.kernel_size)]

        half1, half2 = torch.chunk(guide, 2, dim=1)
        output = torch.cat((half1, guide_mid, half2), dim=1)
        return output


def kernel_trans(kernel, weight):
    '''可以把wieght卷积核平移一步'''
    kernel_size = int(math.sqrt(kernel.size()[1]))
    kernel = F.conv2d(kernel, weight, stride=1, padding=int((kernel_size - 1) / 2))
    return kernel


class CSPNAccelerate(nn.Module):

    def __init__(self, kernel_size, dilation=1, padding=1, stride=1):
        super(CSPNAccelerate, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, kernel, input, input0):  # with standard CSPN, an addition input0 port is added
        bs = input.size()[0]
        h, w = input.size()[2], input.size()[3]
        input_im2col = F.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride)
        kernel = kernel.reshape(bs, self.kernel_size * self.kernel_size, h * w)

        #  standard CSPN
        input0 = input0.view(bs, 1, h * w)
        mid_index = int((self.kernel_size * self.kernel_size - 1) / 2)
        input_im2col[:, mid_index:mid_index + 1, :] = input0

        # print(input_im2col.size(), kernel.size())
        output = torch.einsum('ijk,ijk->ik', (input_im2col, kernel))
        return output.view(bs, 1, h, w)
