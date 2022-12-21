from sparse2dense.dd_branch import get_model as DDBranch
from sparse2dense.mid_branch import get_model as MIDBranch
from pytorch_lightning import LightningModule
import torch


class get_model(LightningModule):

    def __init__(self, args):
        super(get_model, self).__init__()
        self.args = args
        with torch.no_grad():
            self.mid_branch = MIDBranch(args).to('cuda')
            self.dd_branch = DDBranch(args).to('cuda')

    def forward(self, input):
        d = input['velodyne_proj_img'].float()
        input['d'] = d
        input['position'] = input['position'].float()
        input['K'] = input['K'].float()
        with torch.no_grad():
            input = self.mid_branch(input)
            input = self.dd_branch(input)
        return input
