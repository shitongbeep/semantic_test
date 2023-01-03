from sparse2dense._2dpaenet import get_model as _2dpaenet_backbone
from sparse2dense.basic_block import weights_init, DySPN
from pytorch_lightning import LightningModule
import torch


class get_model(LightningModule):

    def __init__(self, args):
        super(get_model, self).__init__()
        self.args = args
        with torch.no_grad():
            self.backbone = _2dpaenet_backbone(args)
        self.dd_dyspn = DySPN(self.args.feat_layer, 7, 6)

        weights_init(self)

    def forward(self, input):
        d = []
        for i in range(4):
            d.append(input['velodyne_proj_img' + str(i)])
        d = torch.cat(d, dim=0)
        with torch.no_grad():
            input = self.backbone(input)

        if 'fust_output' in input.keys():
            coarse_depth = input['fuse_output']
        else:
            coarse_depth = input['dd_branch_output']
        dd_feature = input['dd_feature']
        refined_depth = self.dd_dyspn(dd_feature, coarse_depth, d)
        input['refined_depth'] = refined_depth
        all_refined_depth = []
        for i in range(4):
            all_refined_depth.append(input['refined_depth'][i:i + 1, ...])
        all_refined_depth = torch.cat(all_refined_depth, dim=-1)
        input['all_refined_depth'] = all_refined_depth

        return input
