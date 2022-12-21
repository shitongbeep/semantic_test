from sparse2dense.basic_block import convbnlrelui, deconvbnlrelui, weights_init
from sparse2dense.basic_block import BasicBlockGeo, GeometryFeature, SparseDownSampleClose
import torch.nn as nn
import torch
import pytorch_lightning as pl


class get_model(pl.LightningModule):

    def __init__(self, args):
        super(get_model, self).__init__()
        self.args = args
        self.geoplanes = 3
        self.geofeature = GeometryFeature()

        # *mid_branch network encoder
        self.mid_branch_conv_init = convbnlrelui(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.mid_branch_encoder_layer1 = BasicBlockGeo(inplanes=32, planes=64, stride=2, geoplanes=self.geoplanes)
        self.mid_branch_encoder_layer2 = BasicBlockGeo(inplanes=64, planes=128, stride=2, geoplanes=self.geoplanes)
        self.mid_branch_encoder_layer3 = BasicBlockGeo(inplanes=128, planes=256, stride=2, geoplanes=self.geoplanes)
        self.mid_branch_encoder_layer4 = BasicBlockGeo(inplanes=256, planes=512, stride=2, geoplanes=self.geoplanes)
        self.mid_branch_encoder_layer5 = BasicBlockGeo(inplanes=512, planes=1024, stride=2, geoplanes=self.geoplanes)
        # mid_branch network decoder
        self.mid_branch_decoder_layer4 = deconvbnlrelui(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.mid_branch_decoder_layer3 = deconvbnlrelui(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.mid_branch_decoder_layer2 = deconvbnlrelui(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.mid_branch_decoder_layer1 = deconvbnlrelui(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.mid_branch_conv_uninit = deconvbnlrelui(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.mid_branch_output = convbnlrelui(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)

        # *others
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.sparsepooling = SparseDownSampleClose(stride=2)

        weights_init(self)

    def forward(self, input):
        d = input['d']
        position = input['position']
        K = input['K']

        # *生成几何特征通道
        unorm = position[:, 0:1, :, :]  # 沿着x方向 -1 -> 1
        vnorm = position[:, 1:2, :, :]  # 沿着y方向 1 -> -1
        new_shape = (unorm.shape[0], 1, 1, 1)
        f352 = K[:, 1, 1]
        f352 = f352.view(new_shape)
        c352 = K[:, 1, 2]
        c352 = c352.view(new_shape)
        f1216 = K[:, 0, 0]
        f1216 = f1216.view(new_shape)
        c1216 = K[:, 0, 2]
        c1216 = c1216.view(new_shape)
        # 坐标降采样
        vnorm_s2 = self.pooling(vnorm)
        vnorm_s3 = self.pooling(vnorm_s2)
        vnorm_s4 = self.pooling(vnorm_s3)
        vnorm_s5 = self.pooling(vnorm_s4)
        vnorm_s6 = self.pooling(vnorm_s5)
        unorm_s2 = self.pooling(unorm)
        unorm_s3 = self.pooling(unorm_s2)
        unorm_s4 = self.pooling(unorm_s3)
        unorm_s5 = self.pooling(unorm_s4)
        unorm_s6 = self.pooling(unorm_s5)
        # sparse depth 降采样保留近处点
        valid_mask = torch.where(d > 0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))
        d_s2, vm_s2 = self.sparsepooling(d, valid_mask)
        d_s3, vm_s3 = self.sparsepooling(d_s2, vm_s2)
        d_s4, vm_s4 = self.sparsepooling(d_s3, vm_s3)
        d_s5, vm_s5 = self.sparsepooling(d_s4, vm_s4)
        d_s6, vm_s6 = self.sparsepooling(d_s5, vm_s5)

        geo_s1 = self.geofeature(d, vnorm, unorm, 352, 1216, c352, c1216, f352, f1216)
        geo_s2 = self.geofeature(d_s2, vnorm_s2, unorm_s2, 352, 1216, c352, c1216, f352, f1216)
        geo_s3 = self.geofeature(d_s3, vnorm_s3, unorm_s3, 352, 1216, c352, c1216, f352, f1216)
        geo_s4 = self.geofeature(d_s4, vnorm_s4, unorm_s4, 352, 1216, c352, c1216, f352, f1216)
        geo_s5 = self.geofeature(d_s5, vnorm_s5, unorm_s5, 352, 1216, c352, c1216, f352, f1216)
        geo_s6 = self.geofeature(d_s6, vnorm_s6, unorm_s6, 352, 1216, c352, c1216, f352, f1216)

        # *Encoder
        # 1 --> 32
        mid_branch_feature = self.mid_branch_conv_init(d)
        # iv  32 --> 64
        mid_branch_feature1 = self.mid_branch_encoder_layer1(mid_branch_feature, geo_s1, geo_s2)
        # iii  64 --> 128
        mid_branch_feature2 = self.mid_branch_encoder_layer2(mid_branch_feature1, geo_s2, geo_s3)
        # ii  128 --> 256
        mid_branch_feature3 = self.mid_branch_encoder_layer3(mid_branch_feature2, geo_s3, geo_s4)
        # i  256 --> 512
        mid_branch_feature4 = self.mid_branch_encoder_layer4(mid_branch_feature3, geo_s4, geo_s5)
        mid_branch_feature5 = self.mid_branch_encoder_layer5(mid_branch_feature4, geo_s5, geo_s6)
        # *Decoder
        mid_branch_feature_decoder4 = self.mid_branch_decoder_layer4(mid_branch_feature5)
        mid_branch_feature_decoder4 = mid_branch_feature_decoder4 + mid_branch_feature4
        # 512 --> 256
        mid_branch_feature_decoder3 = self.mid_branch_decoder_layer3(mid_branch_feature4)
        mid_branch_feature_decoder3 = mid_branch_feature_decoder3 + mid_branch_feature3
        # 56 --> 128
        mid_branch_feature_decoder2 = self.mid_branch_decoder_layer2(mid_branch_feature_decoder3)
        mid_branch_feature_decoder2 = mid_branch_feature_decoder2 + mid_branch_feature2
        # 128 --> 64
        mid_branch_feature_decoder1 = self.mid_branch_decoder_layer1(mid_branch_feature_decoder2)
        mid_branch_feature_decoder1 = mid_branch_feature_decoder1 + mid_branch_feature1
        # 64 --> 32
        mid_branch_feature_decoder = self.mid_branch_conv_uninit(mid_branch_feature_decoder1)
        mid_branch_feature_decoder = mid_branch_feature_decoder + mid_branch_feature
        # 32 --> 1
        mid_branch_output = self.mid_branch_output(mid_branch_feature_decoder)

        input['mid_feature_decoder4'] = mid_branch_feature_decoder4
        input['mid_feature_decoder3'] = mid_branch_feature_decoder3
        input['mid_feature_decoder2'] = mid_branch_feature_decoder2
        input['mid_feature_decoder1'] = mid_branch_feature_decoder1
        input['mid_feature_decoder0'] = mid_branch_feature_decoder

        input['geo_s1'] = geo_s1
        input['geo_s2'] = geo_s2
        input['geo_s3'] = geo_s3
        input['geo_s4'] = geo_s4
        input['geo_s5'] = geo_s5
        input['geo_s6'] = geo_s6

        input['mid_branch_output'] = mid_branch_output[:, 0:1, ...]
        input['mid_branch_confidence'] = mid_branch_output[:, 1:2, ...]

        return input
