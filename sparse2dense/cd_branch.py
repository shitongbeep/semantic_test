import torch
from sparse2dense.basic_block import convbnlrelui, deconvbnlrelui, weights_init
from sparse2dense.basic_block import BasicBlockGeo, GeometryFeature, SparseDownSampleClose, RGB2DepthLeanerBlock
import pytorch_lightning as pl
import torch.nn as nn


class get_model(pl.LightningModule):

    def __init__(self, args):
        super(get_model, self).__init__()
        self.save_hyperparameters()
        self.args = args
        self.geoplanes = 3
        self.geofeature = GeometryFeature()

        # *cd_branch network encoder
        self.cd_branch_conv_init = convbnlrelui(in_channels=10, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.cd_branch_encoder_layer1 = BasicBlockGeo(inplanes=32, planes=64, stride=2, geoplanes=self.geoplanes)
        self.cd_branch_encoder_layer2 = BasicBlockGeo(inplanes=64, planes=128, stride=2, geoplanes=self.geoplanes)
        self.cd_branch_encoder_layer3 = BasicBlockGeo(inplanes=128, planes=256, stride=2, geoplanes=self.geoplanes)
        self.cd_branch_encoder_layer4 = BasicBlockGeo(inplanes=256, planes=512, stride=2, geoplanes=self.geoplanes)
        # cd_branch network decoder
        self.cd_branch_decoder_layer3 = deconvbnlrelui(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.cd_branch_decoder_layer2 = deconvbnlrelui(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.cd_branch_decoder_layer1 = deconvbnlrelui(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.cd_branch_conv_uninit = deconvbnlrelui(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.cd_branch_output = convbnlrelui(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        # *others
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.sparsepooling = SparseDownSampleClose(stride=2)
        self.rgbleaner = RGB2DepthLeanerBlock()

        weights_init(self)

    def forward(self, input):
        d = input['d']
        geo_s1 = input['geo_s1']
        geo_s2 = input['geo_s2']
        geo_s3 = input['geo_s3']
        geo_s4 = input['geo_s4']
        geo_s5 = input['geo_s5']
        rgb = input['rgb'].detach().clone()
        rgb = self.rgbleaner(rgb)
        rgbd = torch.cat([rgb, d], dim=1)
        # *Encoder
        # 1 --> 32
        cd_branch_feature = self.cd_branch_conv_init(rgbd)
        # iv  32 --> 64
        cd_branch_feature1 = self.cd_branch_encoder_layer1(cd_branch_feature, geo_s1, geo_s2)
        # iii  64 --> 128
        cd_branch_feature2 = self.cd_branch_encoder_layer2(cd_branch_feature1, geo_s2, geo_s3)
        # ii  128 --> 256
        cd_branch_feature3 = self.cd_branch_encoder_layer3(cd_branch_feature2, geo_s3, geo_s4)
        # i  256 --> 512
        cd_branch_feature4 = self.cd_branch_encoder_layer4(cd_branch_feature3, geo_s4, geo_s5)
        # *Decoder
        # 512 --> 256
        cd_branch_feature_decoder3 = self.cd_branch_decoder_layer3(cd_branch_feature4)
        cd_branch_feature_decoder3 = cd_branch_feature_decoder3 + cd_branch_feature3
        # 256 --> 128
        cd_branch_feature_decoder2 = self.cd_branch_decoder_layer2(cd_branch_feature_decoder3)
        cd_branch_feature_decoder2 = cd_branch_feature_decoder2 + cd_branch_feature2
        # 128 --> 64
        cd_branch_feature_decoder1 = self.cd_branch_decoder_layer1(cd_branch_feature_decoder2)
        cd_branch_feature_decoder1 = cd_branch_feature_decoder1 + cd_branch_feature1
        # 64 --> 32
        cd_branch_feature_decoder = self.cd_branch_conv_uninit(cd_branch_feature_decoder1)
        cd_branch_feature_decoder = cd_branch_feature_decoder + cd_branch_feature
        # 32 --> 1
        cd_branch_output = self.cd_branch_output(cd_branch_feature_decoder)

        input['cd_branch_output'] = cd_branch_output

        return input
