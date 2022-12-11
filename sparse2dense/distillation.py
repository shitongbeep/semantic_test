import torch.nn as nn
import pytorch_lightning as pl
from utils.criteria import FeatureDistance, MaskedMSELoss
from sparse2dense.basic_block import deconvbnlrelui, DepthLeanerBlock, convbnlrelui, weights_init
from torch.nn.parameter import Parameter
import torch


class get_model(pl.LightningModule):

    def __init__(self, args):
        super(get_model, self).__init__()
        self.args = args

        self.distance = FeatureDistance()
        self.loss = MaskedMSELoss()
        self.mid_output = convbnlrelui(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.edgeGx = nn.Conv2d(1, 1, 3, padding=1, padding_mode='replicate')
        self.edgeGy = nn.Conv2d(1, 1, 3, padding=1, padding_mode='replicate')
        edgeweightGx = torch.tensor([-1, 0, 1, -2, 0, 2, -1, 0, 1]).view((3, 3)).unsqueeze(0).unsqueeze(0).float()
        edgeweightGy = torch.tensor([-1, 0, 1, -2, 0, 2, -1, 0, 1]).view((3, 3)).unsqueeze(0).unsqueeze(0).float()
        self.edgeGx.weight = Parameter(edgeweightGx, requires_grad=False)
        self.edgeGy.weight = Parameter(edgeweightGy, requires_grad=False)
        self.edgeGx.weight.requires_grad = False
        self.edgeGy.weight.requires_grad = False
        weights_init(self)

    def forward(self, input):
        gt = input['gt']
        cd_branch_output = input['cd_branch_output'].detach()
        # mid_branch_output = input['mid_branch_output']
        mid_feature = input['mid_feature_decoder0']
        mid_output = self.mid_output(mid_feature)

        cd_edge_x = self.edgeGx(cd_branch_output)
        cd_edge_y = self.edgeGy(cd_branch_output)
        mid_edge_x = self.edgeGx(mid_output)
        mid_edge_y = self.edgeGy(mid_output)

        input['cd_edge_x'] = cd_edge_x
        input['cd_edge_y'] = cd_edge_y
        input['mid_edge_x'] = mid_edge_x
        input['mid_edge_y'] = mid_edge_y

        distance = self.distance(cd_edge_x, mid_edge_x)
        distance += self.distance(cd_edge_y, mid_edge_y)
        loss = self.loss(mid_output, gt)

        input['distance'] = distance
        input['distillation_loss'] = loss

        return input
