from depth_completion.dd_branch import get_model as DDBranch
from pytorch_lightning import LightningModule
import torch


class get_model(LightningModule):
    def __init__(self, args):
        super(get_model, self).__init__()
        self.args = args
        self.dd_branch = DDBranch(args).to('cuda')

    def forward(self, input):
        input = self.dd_branch(input)
        return input
