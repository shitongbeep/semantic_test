from depth_completion.basic_block import convbnrelu, deconvbnrelu
from depth_completion.basic_block import BasicBlockGeo, GeometryFeature, SparseDownSampleClose, SparseDownSample
import torch.nn as nn
import torch
import pytorch_lightning as pl


class get_model(pl.LightningModule):

    def __init__(self, args):
        super(get_model, self).__init__()
        self.save_hyperparameters()
        self.args = args
        self.geoplanes = 3
        self.geofeature = GeometryFeature()

        # *dd_branch network encoder
        self.dd_branch_conv_init = convbnrelu(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.dd_branch_encoder_layer1 = BasicBlockGeo(inplanes=32, planes=64, stride=2, geoplanes=self.geoplanes)
        self.dd_branch_encoder_layer2 = BasicBlockGeo(inplanes=64, planes=128, stride=2, geoplanes=self.geoplanes)
        self.dd_branch_encoder_layer3 = BasicBlockGeo(inplanes=128, planes=256, stride=2, geoplanes=self.geoplanes)
        self.dd_branch_encoder_layer4 = BasicBlockGeo(inplanes=256, planes=512, stride=2, geoplanes=self.geoplanes)
        # dd_branch network decoder
        self.dd_branch_decoder_layer4 = deconvbnrelu(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2, output_padding=0)
        self.dd_branch_decoder_layer3 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dd_branch_decoder_layer2 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dd_branch_decoder_layer1 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dd_branch_conv_uninit = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dd_branch_output = convbnrelu(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        # *others
        self.geo_pooling = SparseDownSample(kernel_size=2, stride=2)
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.sparsepooling = SparseDownSampleClose(stride=2)

    def forward(self, input):
        # geo_s1 = input['velodyne_geo_s1_img0']
        # ds2, geo_s2 = self.geo_pooling(d, geo_s1)
        # ds3, geo_s3 = self.geo_pooling(ds2, geo_s2)
        # ds4, geo_s4 = self.geo_pooling(ds3, geo_s3)
        # _, geo_s5 = self.geo_pooling(ds4, geo_s4)
        position = input['position']
        K = input['K']

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
        unorm_s2 = self.pooling(unorm)
        unorm_s3 = self.pooling(unorm_s2)
        unorm_s4 = self.pooling(unorm_s3)
        unorm_s5 = self.pooling(unorm_s4)
        all_dd_branch_output = []
        for i in range(4):
            d = input['velodyne_proj_img' + str(i)]
            # sparse depth 降采样保留近处点
            valid_mask = torch.where(d > 0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))
            d_s2, vm_s2 = self.sparsepooling(d, valid_mask)
            d_s3, vm_s3 = self.sparsepooling(d_s2, vm_s2)
            d_s4, vm_s4 = self.sparsepooling(d_s3, vm_s3)
            d_s5, vm_s5 = self.sparsepooling(d_s4, vm_s4)
            geo_s1 = self.geofeature(d, vnorm, unorm, 352, 1216, c352, c1216, f352, f1216)
            geo_s2 = self.geofeature(d_s2, vnorm_s2, unorm_s2, 352, 1216, c352, c1216, f352, f1216)
            geo_s3 = self.geofeature(d_s3, vnorm_s3, unorm_s3, 352, 1216, c352, c1216, f352, f1216)
            geo_s4 = self.geofeature(d_s4, vnorm_s4, unorm_s4, 352, 1216, c352, c1216, f352, f1216)
            geo_s5 = self.geofeature(d_s5, vnorm_s5, unorm_s5, 352, 1216, c352, c1216, f352, f1216)

            # *Encoder
            # 1 --> 32
            dd_branch_feature = self.dd_branch_conv_init(d)
            # iv  32 --> 64
            dd_branch_feature1 = self.dd_branch_encoder_layer1(dd_branch_feature, geo_s1, geo_s2)
            # iii  64 --> 128
            dd_branch_feature2 = self.dd_branch_encoder_layer2(dd_branch_feature1, geo_s2, geo_s3)
            # ii  128 --> 256
            dd_branch_feature3 = self.dd_branch_encoder_layer3(dd_branch_feature2, geo_s3, geo_s4)
            # i  256 --> 512
            dd_branch_feature4 = self.dd_branch_encoder_layer4(dd_branch_feature3, geo_s4, geo_s5)
            # *Decoder
            # 1  512 --> 512
            _dd_branch_feature_decoder4 = self.dd_branch_decoder_layer4(dd_branch_feature4)
            dd_branch_feature_decoder4 = _dd_branch_feature_decoder4 + dd_branch_feature4
            # 2  512 --> 256
            _dd_branch_feature_decoder3 = self.dd_branch_decoder_layer3(dd_branch_feature_decoder4)
            dd_branch_feature_decoder3 = _dd_branch_feature_decoder3 + dd_branch_feature3
            # 3  256 --> 128
            _dd_branch_feature_decoder2 = self.dd_branch_decoder_layer2(dd_branch_feature_decoder3)
            dd_branch_feature_decoder2 = _dd_branch_feature_decoder2 + dd_branch_feature2
            # 4  128 --> 64
            _dd_branch_feature_decoder1 = self.dd_branch_decoder_layer1(dd_branch_feature_decoder2)
            dd_branch_feature_decoder1 = _dd_branch_feature_decoder1 + dd_branch_feature1
            # 64 --> 32
            _dd_branch_feature_decoder = self.dd_branch_conv_uninit(dd_branch_feature_decoder1)
            dd_branch_feature_decoder = _dd_branch_feature_decoder + dd_branch_feature
            # 32 --> 1
            dd_branch_output = self.dd_branch_output(dd_branch_feature_decoder)

            input['feature_s1_img' + str(i)] = dd_branch_feature_decoder
            input['feature_s2_img' + str(i)] = dd_branch_feature_decoder1
            input['feature_s3_img' + str(i)] = dd_branch_feature_decoder2

            input['dd_branch_output_img' + str(i)] = dd_branch_output
            all_dd_branch_output.append(dd_branch_output)
        all_dd_branch_output = torch.cat(all_dd_branch_output, dim=-1)
        input['all_dd_branch_output']=all_dd_branch_output

        return input
