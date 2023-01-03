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
        d = []
        for i in range(4):
            d.append(input['velodyne_proj_img' + str(i)])
        d = torch.cat(d, dim=0)
        input['d'] = d
        input['position'] = input['position'].repeat(4, 1, 1, 1)
        input['K'] = input['K'].repeat(4, 1, 1)
        with torch.no_grad():
            input = self.mid_branch(input)
            input = self.dd_branch(input)
        all_fuse_output = []
        for i in range(4):
            if 'fust_output' in input.keys():
                all_fuse_output.append(input['fuse_output'][i:i + 1, ...])
            else:
                all_fuse_output.append(input['dd_branch_output'][i:i + 1, ...])
        all_fuse_output = torch.cat(all_fuse_output, dim=-1)
        input['all_fuse_output'] = all_fuse_output
        return input
