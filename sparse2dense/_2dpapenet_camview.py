from sparse2dense._2dpaenet_camview import get_model as _2dpaenet_backbone
from sparse2dense.basic_block import weights_init, DySPN
from pytorch_lightning import LightningModule
import torch


class get_model(LightningModule):

    def __init__(self, args):
        super(get_model, self).__init__()
        self.args = args
        with torch.no_grad():
            self.backbone = _2dpaenet_backbone(args)
        self.dd_dyspn = DySPN(32, 7, 6)

        weights_init(self)

    def forward(self, input):
        d = input['velodyne_proj_img'].float()
        with torch.no_grad():
            input = self.backbone(input)

        coarse_depth = input['fuse_output']
        dd_feature = input['dd_feature']
        refined_depth = self.dd_dyspn(dd_feature, coarse_depth, d)
        input['refined_depth'] = refined_depth

        return input
