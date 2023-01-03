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

        # *dd_branch network encoder
        self.dd_branch_conv_init = convbnlrelui(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.dd_branch_encoder_layer1 = BasicBlockGeo(inplanes=32, planes=64, stride=2, geoplanes=self.geoplanes)
        self.dd_branch_encoder_layer11 = BasicBlockGeo(inplanes=64, planes=64, stride=1, geoplanes=self.geoplanes)
        self.dd_branch_encoder_layer2 = BasicBlockGeo(inplanes=64, planes=128, stride=2, geoplanes=self.geoplanes)
        self.dd_branch_encoder_layer22 = BasicBlockGeo(inplanes=128, planes=128, stride=1, geoplanes=self.geoplanes)
        self.dd_branch_encoder_layer3 = BasicBlockGeo(inplanes=128, planes=256, stride=2, geoplanes=self.geoplanes)
        self.dd_branch_encoder_layer33 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
        self.dd_branch_encoder_layer4 = BasicBlockGeo(inplanes=256, planes=512, stride=2, geoplanes=self.geoplanes)
        self.dd_branch_encoder_layer44 = BasicBlockGeo(inplanes=512, planes=512, stride=1, geoplanes=self.geoplanes)
        self.dd_branch_encoder_layer5 = BasicBlockGeo(inplanes=512, planes=1024, stride=2, geoplanes=self.geoplanes)
        self.dd_branch_encoder_layer55 = BasicBlockGeo(inplanes=1024, planes=1024, stride=1, geoplanes=self.geoplanes)
        # dd_branch network decoder
        self.dd_branch_decoder_layer4 = deconvbnlrelui(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dd_branch_decoder_layer3 = deconvbnlrelui(in_channels=1024, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dd_branch_decoder_layer2 = deconvbnlrelui(in_channels=512, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dd_branch_decoder_layer1 = deconvbnlrelui(in_channels=256, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dd_branch_conv_uninit = deconvbnlrelui(in_channels=128, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dd_branch_output = convbnlrelui(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

        # *others
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.sparsepooling = SparseDownSampleClose(stride=2)
        self.softmax = nn.Softmax(dim=1)

        weights_init(self)

    def forward(self, input):
        d = input['d']

        geo_s1 = input['geo_s1']
        geo_s2 = input['geo_s2']
        geo_s3 = input['geo_s3']
        geo_s4 = input['geo_s4']
        geo_s5 = input['geo_s5']
        geo_s6 = input['geo_s6']

        # *Encoder
        # 2 --> 32
        dd_branch_feature = self.dd_branch_conv_init(d)
        # 32 --> 64
        dd_branch_feature1 = self.dd_branch_encoder_layer1(dd_branch_feature, geo_s1, geo_s2)
        dd_branch_feature1 = self.dd_branch_encoder_layer11(dd_branch_feature1, geo_s2, geo_s2)
        # 64 --> 128
        dd_branch_feature2 = self.dd_branch_encoder_layer2(dd_branch_feature1, geo_s2, geo_s3)
        dd_branch_feature2 = self.dd_branch_encoder_layer22(dd_branch_feature2, geo_s3, geo_s3)
        # 128 --> 256
        dd_branch_feature3 = self.dd_branch_encoder_layer3(dd_branch_feature2, geo_s3, geo_s4)
        dd_branch_feature3 = self.dd_branch_encoder_layer33(dd_branch_feature3, geo_s4, geo_s4)
        # 256 --> 512
        dd_branch_feature4 = self.dd_branch_encoder_layer4(dd_branch_feature3, geo_s4, geo_s5)
        dd_branch_feature4 = self.dd_branch_encoder_layer44(dd_branch_feature4, geo_s5, geo_s5)
        # 512 --> 1024
        dd_branch_feature5 = self.dd_branch_encoder_layer5(dd_branch_feature4, geo_s5, geo_s6)
        dd_branch_feature5 = self.dd_branch_encoder_layer55(dd_branch_feature5, geo_s6, geo_s6)
        # *Decoder
        # 1024 --> 512
        dd_branch_feature_decoder4 = self.dd_branch_decoder_layer4(dd_branch_feature5)
        dd_branch_feature_decoder4 = torch.cat([dd_branch_feature_decoder4, dd_branch_feature4], dim=1)
        # 1024 --> 256
        dd_branch_feature_decoder3 = self.dd_branch_decoder_layer3(dd_branch_feature_decoder4)
        dd_branch_feature_decoder3 = torch.cat([dd_branch_feature_decoder3, dd_branch_feature3], dim=1)
        # 512 --> 128
        dd_branch_feature_decoder2 = self.dd_branch_decoder_layer2(dd_branch_feature_decoder3)
        dd_branch_feature_decoder2 = torch.cat([dd_branch_feature_decoder2, dd_branch_feature2], dim=1)
        # 256 --> 64
        dd_branch_feature_decoder1 = self.dd_branch_decoder_layer1(dd_branch_feature_decoder2)
        dd_branch_feature_decoder1 = torch.cat([dd_branch_feature_decoder1, dd_branch_feature1], dim=1)
        # 128 --> 32
        dd_branch_feature_decoder = self.dd_branch_conv_uninit(dd_branch_feature_decoder1)
        dd_branch_feature_decoder = torch.cat([dd_branch_feature_decoder, dd_branch_feature], dim=1)
        # 64 --> 1
        dd_branch_output = self.dd_branch_output(dd_branch_feature_decoder)

        input['dd_feature'] = dd_branch_feature_decoder
        input['dd_branch_output'] = dd_branch_output[:, 0:1, ...]

        return input
