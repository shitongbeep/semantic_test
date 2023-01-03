from dataloaders.pc_dataset import SemanticKITTI
import argparse
import yaml
from easydict import EasyDict
from PIL import Image
from torchvision.transforms import transforms
import numpy as np
from torch.utils.data import DataLoader
from sparse2dense._2dpapenet import get_model as DepthCompletionModel
from matplotlib import cm
import torch

totensor = transforms.ToTensor()
topil = transforms.ToPILImage()

args = argparse.ArgumentParser()
args = args.parse_args(args=[])
with open('./config/semantic.yaml') as stream:
    config = yaml.safe_load(stream)
config.update(vars(args))
args = EasyDict(config)
dataset = SemanticKITTI(args=args)
dataloader = DataLoader(dataset, 1, False)

model = DepthCompletionModel(args).cuda()
model = model.load_from_checkpoint('/root/autodl-nas/sparse2dense_s2/best_2dpapenet.ckpt', args=args, strict=False).cuda()

for frame in range(len(dataloader)):
    cur_data = next(iter(dataloader))
    H = cur_data['velodyne_proj_img0'].shape[2]
    W = cur_data['velodyne_proj_img0'].shape[3]
    with torch.no_grad():
        output_data = model(cur_data)
    dense_img = output_data['all_refined_depth'].permute(0, 2, 3, 1).cpu().detach().numpy().squeeze()
    ''' matrix '''
    K = output_data['K'][0].cpu().numpy().squeeze()
    T_velo2img = output_data['T_velo2img'].cpu().numpy().squeeze()[:3, :3]
    T_4img = output_data['T_4img'].cpu().numpy().squeeze()[:3, :3]
    T_rot = output_data['T_rot'].cpu().numpy().squeeze()[:3, :3]
    K_inv = K[:3, :3]
    K_inv = np.linalg.inv(K_inv)
    T_velo2img_inv = np.linalg.inv(T_velo2img)
    T_4img_inv = np.linalg.inv(T_4img)
    T_rot_inv = np.linalg.inv(T_rot)
    ''' point index in depth img '''
    coordinate = np.indices((H, W)).reshape((2, -1))

    for img_idx in range(4):
        z_axis = dense_img[coordinate[0], coordinate[1] + img_idx * W]
        insert_coords = denser_coordinate_lines[i]
        cat_ones = np.ones((insert_coords.shape[0], 1))
        insert_proj_points = np.concatenate([insert_coords, cat_ones], axis=1)
        # -pi/4  pi/4
        z_axis = dense_img[insert_coords[:, 1], insert_coords[:, 0]]
        z_axis = np.expand_dims(z_axis, -1)
        insert_proj_points0 = (K_inv @ insert_proj_points.T).T * z_axis
        mask = insert_proj_points0[:, 2] > 0
        insert_proj_points0 = insert_proj_points0[mask]
        insert_proj_points0 = (T_rot_inv @ T_velo2img_inv @ insert_proj_points0.T).T

        # pi/4  3pi/4
        z_axis = dense_img[insert_coords[:, 1], insert_coords[:, 0] + W]
        z_axis = np.expand_dims(z_axis, -1)
        insert_proj_points1 = (K_inv @ insert_proj_points.T).T * z_axis
        mask = insert_proj_points1[:, 2] > 0
        insert_proj_points1 = insert_proj_points1[mask]
        insert_proj_points1 = (T_rot_inv @ T_4img_inv @ T_velo2img_inv @ insert_proj_points1.T).T
        # 3pi/4  -3pi/4
        z_axis = dense_img[insert_coords[:, 1], insert_coords[:, 0] + 2 * W]
        z_axis = np.expand_dims(z_axis, -1)
        insert_proj_points2 = (K_inv @ insert_proj_points.T).T * z_axis
        mask = insert_proj_points2[:, 2] > 0
        insert_proj_points2 = insert_proj_points2[mask]
        insert_proj_points2 = (T_rot_inv @ T_4img_inv @ T_4img_inv @ T_velo2img_inv @ insert_proj_points2.T).T
        # -3pi/4  -pi/4
        z_axis = dense_img[insert_coords[:, 1], insert_coords[:, 0] + 3 * W]
        z_axis = np.expand_dims(z_axis, -1)
        insert_proj_points3 = (K_inv @ insert_proj_points.T).T * z_axis
        mask = insert_proj_points3[:, 2] > 0
        insert_proj_points3 = insert_proj_points3[mask]
        insert_proj_points3 = (T_rot_inv @ T_4img_inv @ T_4img_inv @ T_4img_inv @ T_velo2img_inv @ insert_proj_points3.T).T

        insert_proj_points = [insert_proj_points0, insert_proj_points1, insert_proj_points2, insert_proj_points3]
        insert_proj_points = np.concatenate(insert_proj_points, axis=0)
        # print(insert_proj_points.shape)
        # print(insert_proj_points3.shape)
        # sparse_proj_points.insert(i, insert_proj_points)
        # sparse_proj_points.append(insert_proj_points)
        sparse_proj_points.insert(2 * i + 1, insert_proj_points)
        # print(insert_proj_points.shape)

    sparse_proj_points = np.concatenate(sparse_proj_points, axis=0)
    ''' transform points to velodyne_axis '''
