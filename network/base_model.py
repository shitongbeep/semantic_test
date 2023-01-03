#!/usr/bin/env python
# encoding: utf-8
'''
@author: Shi Tong
@file: base_model.py
@time: 2022/11/9 14:12
'''
import torch
import os
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from utils.criteria import MaskedMSELoss, MaskedL1Loss, Distance
from utils.vis_utils import save_depth_as_uint16png_upload, save_depth_as_uint8colored
from utils.metrics import AverageMeter
from utils.logger import logger
from typing import Any, Dict


class LightningBaseModel(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.depth_criterion = MaskedMSELoss() if (args.criterion == 'l2') else MaskedL1Loss()
        self.distance = Distance()
        self.dd_average_meter = AverageMeter()
        self.cd_average_meter = AverageMeter()
        self.refine_average_meter = AverageMeter()
        self.mylogger = logger(args)

    def configure_optimizers(self):
        # *optimizer
        if self.args.network_model == '_2dpaenet':
            # 训练没有CSPN++部分的backbone
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay, betas=(0.9, 0.99))
        elif self.args.network_model == '_2dpapenet' and not self.args.freeze_backbone:
            # 训练带有CSPN++的整个网络
            model_bone_params = [p for _, p in self.backbone.named_parameters() if p.requires_grad]
            model_new_params = [p for _, p in self.named_parameters() if p.requires_grad]
            model_new_params = list(set(model_new_params) - set(model_bone_params))
            optimizer = torch.optim.Adam([{
                'params': model_bone_params,
                'lr': self.args.learning_rate
            }, {
                'params': model_new_params
            }],
                                         lr=self.args.learning_rate,
                                         weight_decay=self.args.weight_decay,
                                         betas=(0.9, 0.99))
        elif self.args.network_model == '_2dpapenet' and self.args.freeze_backbone:
            # 固定backbone，训练CSPN++
            for p in self.backbone.parameters():
                p.requires_grad = False
            model_named_params = [p for _, p in self.named_parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(model_named_params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay, betas=(0.9, 0.99))
        else:
            # 训练参数错误
            raise NotImplementedError('in base_model.py : optimizer wrong, config network_model and freeze_backbone in .yaml')

        # *lr_scheduler
        if self.args.lr_scheduler == 'StepLR':
            lr_scheduler = StepLR(optimizer, step_size=self.args.decay_step, gamma=self.args.decay_rate)
        elif self.args.lr_scheduler == 'ReduceLROnPlateau':
            lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=self.args.decay_rate, patience=self.args.decay_step, verbose=True)
        elif self.args.lr_scheduler == 'CosineAnnealingLR':
            lr_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.args.epochs - 4,
                eta_min=1e-5,
            )
        else:
            raise NotImplementedError('in base_model.py : lr_scheduler wrong, config lr_scheduler in .yaml')

        scheduler = {'scheduler': lr_scheduler, 'interval': 'epoch', 'frequency': 1}

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }

    def forward(self, data) -> Any:
        pass

    def training_step(self, data: Dict, batch_idx):
        if self.args.network_model == '_2dpaenet':
            data = self.forward(data)
            dd_branch_output = data['dd_branch_output']
            cd_branch_output = data['cd_branch_output']
            gt = data['gt']
            distillation_loss = data['distillation_loss']
            distance = data['distance']
            dd_loss = self.depth_criterion(dd_branch_output, gt)
            cd_loss = self.depth_criterion(cd_branch_output, gt)
            loss = (dd_loss + cd_loss + distillation_loss) / 3 + distance
            self.log('train/loss', loss.item())
            self.log('train/dd_loss', dd_loss.item())
            self.log('train/cd_loss', cd_loss.item())
            self.log('train/distance', distance.item())
            self.log('train/student_loss', data['student_loss'].item())
            self.log('train/teacher_loss', data['teacher_loss'].item())
            return loss
        elif self.args.network_model == '_2dpapenet' and self.args.freeze_backbone:
            data = self.forward(data)
            refined_output = data['refined_depth']
            gt = data['gt']
            loss = self.depth_criterion(refined_output, gt)
            self.log('train/loss', loss.item())
            return loss
        elif self.args.network_model == '_2dpapenet' and not self.args.freeze_backbone:
            pass
        else:
            raise NotImplementedError('in base_model.py : train wrong, config network_model and freeze_backbone in .yaml')

    def validation_step(self, data, batch_idx):
        if self.args.network_model == '_2dpaenet':
            data = self.forward(data)
            dd_branch_output = data['dd_branch_output']
            cd_branch_output = data['cd_branch_output']
            gt = data['gt']

            dd_loss = self.depth_criterion(dd_branch_output, gt)
            cd_loss = self.depth_criterion(cd_branch_output, gt)
            self.dd_average_meter(dd_branch_output, gt)
            self.cd_average_meter(cd_branch_output, gt)

            self.log('val/dd_loss', dd_loss.item())
            self.log('val/cd_loss', cd_loss.item())

            self.mylogger.conditional_save_img_comparison(batch_idx, data, dd_branch_output, self.current_epoch)
            return dd_loss
        elif self.args.network_model == '_2dpapenet' and self.args.freeze_backbone:
            data = self.forward(data)
            refined_output = data['refined_depth']
            gt = data['gt']
            self.refine_average_meter(refined_output, gt)
            loss = self.depth_criterion(refined_output, gt)
            self.log('val/loss', loss.item())
            self.mylogger.conditional_save_img_comparison(batch_idx, data, refined_output, self.current_epoch)
            return loss
        elif self.args.network_model == '_2dpapenet' and not self.args.freeze_backbone:
            pass
        else:
            raise NotImplementedError('in base_model.py : validation wrong, config network_model and freeze_backbone in .yaml')

    def test_step(self, data, batch_idx):
        data = self.forward(data)
        if self.args.network_model == '_2dpaenet':
            pred = data['dd_branch_output']
        elif self.args.network_model == '_2dpapenet':
            pred = data['refined_depth']
        else:
            raise NotImplementedError('in base_model.py : test wrong, wrong network_model: ' + self.args.network_model)
        str_i = str(self.current_epoch * self.args.batch_size + batch_idx)
        path_i = str_i.zfill(10) + '.png'
        path = os.path.join(self.args.data_folder_save, path_i)
        save_depth_as_uint16png_upload(pred, path)
        path_i = str_i.zfill(10) + '.png'
        path = os.path.join(self.args.data_folder_save + 'color/', path_i)
        save_depth_as_uint8colored(pred, path)

    def validation_epoch_end(self, outputs):
        if self.args.network_model == '_2dpaenet':
            self.dd_average_meter.compute()
            self.cd_average_meter.compute()
            self.log('val/rmse', self.dd_average_meter.sum_rmse, on_epoch=True, sync_dist=True)
            self.log('val/dd_rmse', self.dd_average_meter.sum_rmse, on_epoch=True, sync_dist=True)
            self.log('val/cd_rmse', self.cd_average_meter.sum_rmse, on_epoch=True, sync_dist=True)
            str_print = 'Validation : '
            str_print += '\ndd_branch: Current val rmse is %.3f while the best val rmse is %.3f' % (self.dd_average_meter.sum_rmse,
                                                                                                    self.dd_average_meter.best_rmse)
            str_print += '\ncd_branch: Current val rmse is %.3f while the best val rmse is %.3f' % (self.cd_average_meter.sum_rmse,
                                                                                                    self.cd_average_meter.best_rmse)
            self.mylogger.conditional_save_info(self.cd_average_meter, self.current_epoch)
            self.mylogger.conditional_save_info(self.dd_average_meter, self.current_epoch)
            if self.dd_average_meter.best_rmse > self.dd_average_meter.rmse:
                self.mylogger.save_img_comparison_as_best(self.current_epoch)
                self.mylogger.conditional_save_info(self.cd_average_meter, self.current_epoch, True)
                self.mylogger.conditional_save_info(self.dd_average_meter, self.current_epoch, True)
            self.dd_average_meter.reset()
            self.cd_average_meter.reset()
        elif self.args.network_model == '_2dpapenet' and self.args.freeze_backbone:
            self.refine_average_meter.compute()
            self.log('val/rmse', self.refine_average_meter.sum_rmse + self.refine_average_meter.sum_rmse, on_epoch=True, sync_dist=True)
            str_print = 'Validation : '
            str_print += '\nRefine: Current val rmse is %.3f while the best val rmse is %.3f' % (self.refine_average_meter.sum_rmse,
                                                                                                 self.refine_average_meter.best_rmse)
            self.mylogger.conditional_save_info(self.refine_average_meter, self.current_epoch)
            if self.refine_average_meter.best_rmse > self.refine_average_meter.rmse:
                self.mylogger.save_img_comparison_as_best(self.current_epoch)
                self.mylogger.conditional_save_info(self.refine_average_meter, self.current_epoch, True)
            self.refine_average_meter.reset()
        elif self.args.network_model == '_2dpapenet' and not self.args.freeze_backbone:
            pass
        else:
            raise NotImplementedError('in base_model.py : validation end wrong, config network_model and freeze_backbone in .yaml')

    def test_epoch_end(self, outputs):
        pass
        # if self.args.network_model == '_2dpaenet' and self.args.mode == 'val':
        #     self.stu_average_meter.compute()
        #     self.mylogger.conditional_save_info(self.stu_average_meter, self.current_epoch, False)

    def on_after_backward(self) -> None:
        """
        Skipping updates in case of unstable gradients
        https://github.com/Lightning-AI/lightning/issues/4956
        """
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            print(f'detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()
