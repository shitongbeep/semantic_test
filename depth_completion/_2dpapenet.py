from depth_completion._2dpaenet import get_model as _2dpaenet_backbone
from depth_completion.basic_block import weights_init, kernel_trans, convbn, convbnrelu, CSPNAccelerate, CSPNGenerateAccelerate, SparseDownSampleClose, CSPNGenerate, CSPN
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
from pytorch_lightning import LightningModule


class get_model(LightningModule):

    def __init__(self, args):
        super(get_model, self).__init__()
        self.args = args
        self.backbone = _2dpaenet_backbone(args)

        self.kernel_conf_layer_s1 = convbn(32, 3)
        self.mask_layer_s1 = convbn(32, 1)
        self.kernel_conf_layer_s2 = convbn(64, 3)
        self.mask_layer_s2 = convbn(64, 1)
        self.dimhalf_s2 = convbnrelu(64, 32, 1, 1, 0)
        self.att_12 = convbnrelu(64, 2)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.downsample = SparseDownSampleClose(stride=2)
        self.softmax = nn.Softmax(dim=1)
        self.acce = True

        if self.acce:
            self.iter_guide_layer3_s1 = CSPNGenerate(32, 3)
            self.iter_guide_layer5_s1 = CSPNGenerate(32, 5)
            self.iter_guide_layer7_s1 = CSPNGenerate(32, 7)

            self.iter_guide_layer3_s2 = CSPNGenerate(64, 3)
            self.iter_guide_layer5_s2 = CSPNGenerate(64, 5)
            self.iter_guide_layer7_s2 = CSPNGenerate(64, 7)

            self.CSPN3 = CSPN(3)
            self.CSPN5 = CSPN(5)
            self.CSPN7 = CSPN(7)
        else:
            self.iter_guide_layer3_s1 = CSPNGenerateAccelerate(32, 3)
            self.iter_guide_layer5_s1 = CSPNGenerateAccelerate(32, 5)
            self.iter_guide_layer7_s1 = CSPNGenerateAccelerate(32, 7)
            self.CSPN3_s1 = CSPNAccelerate(kernel_size=3, dilation=1, padding=1, stride=1)
            self.CSPN5_s1 = CSPNAccelerate(kernel_size=5, dilation=1, padding=2, stride=1)
            self.CSPN7_s1 = CSPNAccelerate(kernel_size=7, dilation=1, padding=3, stride=1)
            self.iter_guide_layer3_s2 = CSPNGenerateAccelerate(64, 3)
            self.iter_guide_layer5_s2 = CSPNGenerateAccelerate(64, 5)
            self.iter_guide_layer7_s2 = CSPNGenerateAccelerate(64, 7)
            self.CSPN3_s2 = CSPNAccelerate(kernel_size=3, dilation=2, padding=2, stride=1)
            self.CSPN5_s2 = CSPNAccelerate(kernel_size=5, dilation=2, padding=4, stride=1)
            self.CSPN7_s2 = CSPNAccelerate(kernel_size=7, dilation=2, padding=6, stride=1)
            self.nnupsample2 = nn.UpsamplingNearest2d(scale_factor=2)

            def makeEnocder(kernel_size):
                encoder = torch.zeros(kernel_size * kernel_size, kernel_size * kernel_size, kernel_size, kernel_size).cuda()
                kernel_range_list = [i for i in range(kernel_size - 1, -1, -1)]
                ls = []
                for i in range(kernel_size):
                    ls.extend(kernel_range_list)
                index = [[j for j in range(kernel_size * kernel_size - 1, -1, -1)], [j for j in range(kernel_size * kernel_size)],
                         [val for val in kernel_range_list for j in range(kernel_size)], ls]
                encoder[index] = 1
                encoder = Parameter(encoder, requires_grad=False)
                return encoder

            self.encoder3 = makeEnocder(3)
            self.encoder5 = makeEnocder(5)
            self.encoder7 = makeEnocder(7)
        weights_init(self)

    def forward(self, input):
        input = self.backbone(input)
        all_refine_depth = []
        for img_num in range(4):
            d = input['velodyne_proj_img' + str(img_num)]
            valid_mask = torch.where(d > 0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))
            coarse_depth = input['dd_branch_output_img' + str(img_num)]
            feature_s1 = input['feature_s1_img' + str(img_num)]
            feature_s2 = input['feature_s2_img' + str(img_num)]
            depth = coarse_depth

            if self.acce:
                d_s2, valid_mask_s2 = self.downsample(d, valid_mask)
                mask_s2 = self.mask_layer_s2(feature_s2)
                mask_s2 = torch.sigmoid(mask_s2)
                mask_s2 = mask_s2 * valid_mask_s2
                mask_s1 = self.mask_layer_s1(feature_s1)
                mask_s1 = torch.sigmoid(mask_s1)
                mask_s1 = mask_s1 * valid_mask

                kernel_conf_s2 = self.kernel_conf_layer_s2(feature_s2)
                kernel_conf_s2 = self.softmax(kernel_conf_s2)
                kernel_conf3_s2 = kernel_conf_s2[:, 0:1, :, :]
                kernel_conf5_s2 = kernel_conf_s2[:, 1:2, :, :]
                kernel_conf7_s2 = kernel_conf_s2[:, 2:3, :, :]

                kernel_conf_s1 = self.kernel_conf_layer_s1(feature_s1)
                kernel_conf_s1 = self.softmax(kernel_conf_s1)
                kernel_conf3_s1 = kernel_conf_s1[:, 0:1, :, :]
                kernel_conf5_s1 = kernel_conf_s1[:, 1:2, :, :]
                kernel_conf7_s1 = kernel_conf_s1[:, 2:3, :, :]

                feature_12 = torch.cat((feature_s1, self.upsample2(self.dimhalf_s2(feature_s2))), 1)
                att_map_12 = self.softmax(self.att_12(feature_12))

                guide3_s2 = self.iter_guide_layer3_s2(feature_s2)
                guide5_s2 = self.iter_guide_layer5_s2(feature_s2)
                guide7_s2 = self.iter_guide_layer7_s2(feature_s2)
                guide3_s1 = self.iter_guide_layer3_s1(feature_s1)
                guide5_s1 = self.iter_guide_layer5_s1(feature_s1)
                guide7_s1 = self.iter_guide_layer7_s1(feature_s1)

                depth_s2 = depth
                depth_s2_00 = depth_s2[:, :, 0::2, 0::2]
                depth_s2_01 = depth_s2[:, :, 0::2, 1::2]
                depth_s2_10 = depth_s2[:, :, 1::2, 0::2]
                depth_s2_11 = depth_s2[:, :, 1::2, 1::2]

                depth_s2_00_h0 = depth3_s2_00 = depth5_s2_00 = depth7_s2_00 = depth_s2_00
                depth_s2_01_h0 = depth3_s2_01 = depth5_s2_01 = depth7_s2_01 = depth_s2_01
                depth_s2_10_h0 = depth3_s2_10 = depth5_s2_10 = depth7_s2_10 = depth_s2_10
                depth_s2_11_h0 = depth3_s2_11 = depth5_s2_11 = depth7_s2_11 = depth_s2_11

                for i in range(6):
                    depth3_s2_00 = self.CSPN3(guide3_s2, depth3_s2_00, depth_s2_00_h0)
                    depth3_s2_00 = mask_s2 * d_s2 + (1 - mask_s2) * depth3_s2_00
                    depth5_s2_00 = self.CSPN5(guide5_s2, depth5_s2_00, depth_s2_00_h0)
                    depth5_s2_00 = mask_s2 * d_s2 + (1 - mask_s2) * depth5_s2_00
                    depth7_s2_00 = self.CSPN7(guide7_s2, depth7_s2_00, depth_s2_00_h0)
                    depth7_s2_00 = mask_s2 * d_s2 + (1 - mask_s2) * depth7_s2_00

                    depth3_s2_01 = self.CSPN3(guide3_s2, depth3_s2_01, depth_s2_01_h0)
                    depth3_s2_01 = mask_s2 * d_s2 + (1 - mask_s2) * depth3_s2_01
                    depth5_s2_01 = self.CSPN5(guide5_s2, depth5_s2_01, depth_s2_01_h0)
                    depth5_s2_01 = mask_s2 * d_s2 + (1 - mask_s2) * depth5_s2_01
                    depth7_s2_01 = self.CSPN7(guide7_s2, depth7_s2_01, depth_s2_01_h0)
                    depth7_s2_01 = mask_s2 * d_s2 + (1 - mask_s2) * depth7_s2_01

                    depth3_s2_10 = self.CSPN3(guide3_s2, depth3_s2_10, depth_s2_10_h0)
                    depth3_s2_10 = mask_s2 * d_s2 + (1 - mask_s2) * depth3_s2_10
                    depth5_s2_10 = self.CSPN5(guide5_s2, depth5_s2_10, depth_s2_10_h0)
                    depth5_s2_10 = mask_s2 * d_s2 + (1 - mask_s2) * depth5_s2_10
                    depth7_s2_10 = self.CSPN7(guide7_s2, depth7_s2_10, depth_s2_10_h0)
                    depth7_s2_10 = mask_s2 * d_s2 + (1 - mask_s2) * depth7_s2_10

                    depth3_s2_11 = self.CSPN3(guide3_s2, depth3_s2_11, depth_s2_11_h0)
                    depth3_s2_11 = mask_s2 * d_s2 + (1 - mask_s2) * depth3_s2_11
                    depth5_s2_11 = self.CSPN5(guide5_s2, depth5_s2_11, depth_s2_11_h0)
                    depth5_s2_11 = mask_s2 * d_s2 + (1 - mask_s2) * depth5_s2_11
                    depth7_s2_11 = self.CSPN7(guide7_s2, depth7_s2_11, depth_s2_11_h0)
                    depth7_s2_11 = mask_s2 * d_s2 + (1 - mask_s2) * depth7_s2_11

                depth_s2_00 = kernel_conf3_s2 * depth3_s2_00 + kernel_conf5_s2 * depth5_s2_00 + kernel_conf7_s2 * depth7_s2_00
                depth_s2_01 = kernel_conf3_s2 * depth3_s2_01 + kernel_conf5_s2 * depth5_s2_01 + kernel_conf7_s2 * depth7_s2_01
                depth_s2_10 = kernel_conf3_s2 * depth3_s2_10 + kernel_conf5_s2 * depth5_s2_10 + kernel_conf7_s2 * depth7_s2_10
                depth_s2_11 = kernel_conf3_s2 * depth3_s2_11 + kernel_conf5_s2 * depth5_s2_11 + kernel_conf7_s2 * depth7_s2_11

                depth_s2[:, :, 0::2, 0::2] = depth_s2_00
                depth_s2[:, :, 0::2, 1::2] = depth_s2_01
                depth_s2[:, :, 1::2, 0::2] = depth_s2_10
                depth_s2[:, :, 1::2, 1::2] = depth_s2_11

                # feature_12 = torch.cat((feature_s1, self.upsample(self.dimhalf_s2(feature_s2))), 1)
                # att_map_12 = self.softmax(self.att_12(feature_12))
                refined_depth_s2 = depth * att_map_12[:, 0:1, :, :] + depth_s2 * att_map_12[:, 1:2, :, :]
                # refined_depth_s2 = depth

                depth3 = depth5 = depth7 = refined_depth_s2

                # prop
                for i in range(6):
                    depth3 = self.CSPN3(guide3_s1, depth3, depth)
                    depth3 = mask_s1 * d + (1 - mask_s1) * depth3
                    depth5 = self.CSPN5(guide5_s1, depth5, depth)
                    depth5 = mask_s1 * d + (1 - mask_s1) * depth5
                    depth7 = self.CSPN7(guide7_s1, depth7, depth)
                    depth7 = mask_s1 * d + (1 - mask_s1) * depth7

                refined_depth = kernel_conf3_s1 * depth3 + kernel_conf5_s1 * depth5 + kernel_conf7_s1 * depth7
            else:
                d_s2, valid_mask_s2 = self.downsample(d, valid_mask)  # saprse depth, sparse mask 降采样
                mask_s2 = self.mask_layer_s2(feature_s2)
                mask_s2 = torch.sigmoid(mask_s2)
                mask_s2 = mask_s2 * valid_mask_s2  # spaser>0点mask的feature_s2层
                depth_s2 = self.nnupsample2(d_s2)
                mask_s2 = self.nnupsample2(mask_s2)  # 上采样到和coarse_depth相同大小

                kernel_conf_s2 = self.kernel_conf_layer_s2(feature_s2)  # feature_s2生成的3 5 7三种不同dilation的卷积核
                kernel_conf_s2 = self.softmax(kernel_conf_s2)
                kernel_conf3_s2 = self.nnupsample2(kernel_conf_s2[:, 0:1, :, :])
                kernel_conf5_s2 = self.nnupsample2(kernel_conf_s2[:, 1:2, :, :])
                kernel_conf7_s2 = self.nnupsample2(kernel_conf_s2[:, 2:3, :, :])

                mask_s1 = self.mask_layer_s1(feature_s1)
                mask_s1 = torch.sigmoid(mask_s1)
                mask_s1 = mask_s1 * valid_mask  # spaser>0点mask的feature_s1层

                kernel_conf_s1 = self.kernel_conf_layer_s1(feature_s1)  # feature_s1生成的3 5 7三种不同dilation的卷积核
                kernel_conf_s1 = self.softmax(kernel_conf_s1)
                kernel_conf3_s1 = kernel_conf_s1[:, 0:1, :, :]
                kernel_conf5_s1 = kernel_conf_s1[:, 1:2, :, :]
                kernel_conf7_s1 = kernel_conf_s1[:, 2:3, :, :]

                # CSPN Generator 生成kernel_size^2-1维权重层 不同feature和不同卷积核
                guide3_s2 = self.iter_guide_layer3_s2(feature_s2)
                guide5_s2 = self.iter_guide_layer5_s2(feature_s2)
                guide7_s2 = self.iter_guide_layer7_s2(feature_s2)
                guide3_s1 = self.iter_guide_layer3_s1(feature_s1)
                guide5_s1 = self.iter_guide_layer5_s1(feature_s1)
                guide7_s1 = self.iter_guide_layer7_s1(feature_s1)
                # 用encoder将9维权重层分别向8个方向移动一步，形成了卷积核权重层
                guide3_s1 = kernel_trans(guide3_s1, self.encoder3)
                guide5_s1 = kernel_trans(guide5_s1, self.encoder5)
                guide7_s1 = kernel_trans(guide7_s1, self.encoder7)
                guide3_s2 = kernel_trans(guide3_s2, self.encoder3)
                guide5_s2 = kernel_trans(guide5_s2, self.encoder5)
                guide7_s2 = kernel_trans(guide7_s2, self.encoder7)
                # feature_s2 生成的guide需要上采样到和coarse_depth相同大小
                guide3_s2 = self.nnupsample2(guide3_s2)
                guide5_s2 = self.nnupsample2(guide5_s2)
                guide7_s2 = self.nnupsample2(guide7_s2)

                depth3 = depth5 = depth7 = depth
                for i in range(6):
                    depth3 = self.CSPN3_s2(guide3_s2, depth3, coarse_depth)
                    depth3 = mask_s2 * depth_s2 + (1 - mask_s2) * depth3
                    depth5 = self.CSPN5_s2(guide5_s2, depth5, coarse_depth)
                    depth5 = mask_s2 * depth_s2 + (1 - mask_s2) * depth5
                    depth7 = self.CSPN7_s2(guide7_s2, depth7, coarse_depth)
                    depth7 = mask_s2 * depth_s2 + (1 - mask_s2) * depth7

                depth_s2 = kernel_conf3_s2 * depth3 + kernel_conf5_s2 * depth5 + kernel_conf7_s2 * depth7
                refined_depth_s2 = depth_s2
                depth3 = depth5 = depth7 = refined_depth_s2

                # 再用feature_s1生成CSPN guide操作 refine_depth_s2 & coarse_depth --> refine_depth
                for i in range(6):
                    depth3 = self.CSPN3_s1(guide3_s1, depth3, depth)
                    depth3 = mask_s1 * d + (1 - mask_s1) * depth3
                    depth5 = self.CSPN5_s1(guide5_s1, depth5, depth)
                    depth5 = mask_s1 * d + (1 - mask_s1) * depth5
                    depth7 = self.CSPN7_s1(guide7_s1, depth7, depth)
                    depth7 = mask_s1 * d + (1 - mask_s1) * depth7

                refined_depth = kernel_conf3_s1 * depth3 + kernel_conf5_s1 * depth5 + kernel_conf7_s1 * depth7
            input.pop('feature_s1_img' + str(img_num))
            input.pop('feature_s2_img' + str(img_num))
            input.pop('feature_s3_img' + str(img_num))
            input.pop('dd_branch_output_img' + str(img_num))
            input['refined_depth' + str(img_num)] = refined_depth
            all_refine_depth.append(refined_depth)

        all_refine_depth = torch.cat(all_refine_depth, dim=-1)
        input['refined_depth'] = all_refine_depth
        input.pop('all_dd_branch_output')

        return input
