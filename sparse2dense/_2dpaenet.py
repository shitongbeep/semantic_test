from sparse2dense.cd_branch import get_model as CDBranch
from sparse2dense.dd_branch import get_model as DDBranch
from sparse2dense.mid_branch import get_model as MIDBranch
from sparse2dense.distillation import get_model as Fusion
from pytorch_lightning import LightningModule
import torch


class get_model(LightningModule):

    def __init__(self, args, all_branch=False):
        super(get_model, self).__init__()
        self.args = args
        self.all_branch = all_branch
        with torch.no_grad():
            self.mid_branch = MIDBranch(args).to('cuda')
            self.dd_branch = DDBranch(args).to('cuda')
            if self.all_branch:
                self.cd_branch = CDBranch(args).to('cuda')
                self.fusion = Fusion(args).to('cuda')

    def forward(self, input):
        d = []
        for i in range(4):
            d.append(input['velodyne_proj_img' + str(i)])
        d = torch.cat(d, dim=0)
        input['d'] = d
        input['position'] = input['position'].repeat(4, 1, 1, 1)
        input['K'] = input['K'].repeat(4, 1, 1)
        with torch.no_grad():
            input = self.mid_branch(input)
            if self.all_branch:
                input = self.cd_branch(input)
                input = self.fusion(input)
            input = self.dd_branch(input)
            return input
