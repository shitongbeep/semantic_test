import os
import numpy as np
import yaml
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torch
from utils import CoordConv

totensor = transforms.ToTensor()


def absoluteFilePaths(directory, num_vote):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            for _ in range(num_vote):
                yield os.path.abspath(os.path.join(dirpath, f))


def getDenserCoord_bak(pitch, jump, K, H, W, points_num, add_lines=32):
    fai_range = []
    for i in range(jump.max()+1):
        if i == add_lines:
            break
        fai_range.append((np.mean(pitch[jump == i]) + 0.2) / 180 * np.pi)
    denser_mask = [[np.cos(fai) * np.sin(theta), -np.sin(fai), np.cos(fai) * np.cos(theta)]
                   for theta in np.linspace(-np.pi / 4, np.pi / 4, points_num) for fai in fai_range]
    denser_mask = np.array(denser_mask)
    denser_mask = (denser_mask / denser_mask[:, 2:3])
    cat_ones = np.ones((denser_mask.shape[0], 1))
    denser_mask = np.concatenate([denser_mask, cat_ones], axis=1)

    denser_coordinate = ((K @ denser_mask.T).T)[:, :2].astype(np.int32)
    mask = (denser_coordinate[:, 0] >= 0) * (denser_coordinate[:, 0] < W) * (denser_coordinate[:, 1] > 0) * (denser_coordinate[:, 1] < H)
    denser_coordinate = denser_coordinate[mask]
    return denser_coordinate


def getDenserCoord(pitch, jump, K, H, W, points_num, add_lines=32):
    denser_coord_lines = []
    for i in range(jump.max()):
        if i == add_lines:
            break
        fai = (np.mean(pitch[jump == i]) - 0.2) / 180 * np.pi
        cur_line = np.array([[np.cos(fai) * np.sin(theta), -np.sin(fai), np.cos(fai) * np.cos(theta)]
                             for theta in np.linspace(np.pi / 4, -np.pi / 4, points_num)])
        cur_line = cur_line / cur_line[:, 2:3]
        cat_ones = np.ones((cur_line.shape[0], 1))
        cur_line = np.concatenate([cur_line, cat_ones], axis=1)
        cur_coordinate = ((K @ cur_line.T).T)[:, :2].astype(np.int32)
        mask = (cur_coordinate[:, 0] >= 0) * (cur_coordinate[:, 0] < W) * (cur_coordinate[:, 1] > 0) * (cur_coordinate[:, 1] < H)
        cur_coordinate = cur_coordinate[mask]
        denser_coord_lines.append(cur_coordinate)

    denser_coordinate = np.concatenate(denser_coord_lines, axis=0)
    return denser_coordinate, denser_coord_lines


def deal_occlusion(pitch, jump):
    mask = []
    for line_idx in range(jump.max()+1):
        line = pitch[jump == line_idx]
        line_mask = np.ones((line.shape[0], 1))
        max_pitch = line[0]
        for i in range(len(line)):
            if line[i] >= max_pitch:
                max_pitch = line[i]
            else:
                line_mask[i] = 0
        mask.append(line_mask)
    mask = np.concatenate(mask, axis=0).astype(np.bool8).squeeze()
    return mask


def get_velodyne_proj_img(start_rot, proj_points, velodyne_proj_img, H, W, velodyne_proj):
    K = velodyne_proj['K']
    T_4img = velodyne_proj['T_4img']
    T_velo2img = velodyne_proj['T_velo2img']
    proj_points0 = (start_rot[:3, :3] @ proj_points.T).T
    mask = proj_points0[:, 0] > 0
    proj_points0 = proj_points0[mask]
    proj_points0 = (T_velo2img[:3, :3] @ proj_points0.T).T
    # proj_distance0 = proj_distance[mask]
    cat_array = np.ones((proj_points0.shape[0], 1))
    coordinate = np.concatenate([proj_points0, cat_array], axis=1)
    coordinate0 = (K @ coordinate.T).T[:, 0:3]
    coordinate0 = (coordinate0 / coordinate0[:, 2:3])[:, 0:2]

    proj_points1 = (start_rot[:3, :3] @ proj_points.T).T
    mask = proj_points1[:, 1] > 0
    proj_points1 = proj_points1[mask]
    proj_points1 = (T_velo2img[:3, :3] @ T_4img[:3, :3] @ proj_points1.T).T
    # proj_distance1 = proj_distance[mask]
    cat_array = np.ones((proj_points1.shape[0], 1))
    coordinate = np.concatenate([proj_points1, cat_array], axis=1)
    coordinate1 = (K @ coordinate.T).T[:, 0:3]
    coordinate1 = (coordinate1 / coordinate1[:, 2:3])[:, 0:2]

    proj_points2 = (start_rot[:3, :3] @ proj_points.T).T
    mask = proj_points2[:, 0] < 0
    proj_points2 = proj_points2[mask]
    proj_points2 = (T_velo2img[:3, :3] @ T_4img[:3, :3] @ T_4img[:3, :3] @ proj_points2.T).T
    # proj_distance2 = proj_distance[mask]
    cat_array = np.ones((proj_points2.shape[0], 1))
    coordinate = np.concatenate([proj_points2, cat_array], axis=1)
    coordinate2 = (K @ coordinate.T).T[:, 0:3]
    coordinate2 = (coordinate2 / coordinate2[:, 2:3])[:, 0:2]

    proj_points3 = (start_rot[:3, :3] @ proj_points.T).T
    mask = proj_points3[:, 1] < 0
    proj_points3 = proj_points3[mask]
    proj_points3 = (T_velo2img[:3, :3] @ T_4img[:3, :3] @ T_4img[:3, :3] @ T_4img[:3, :3] @ proj_points3.T).T
    # proj_distance3 = proj_distance[mask]
    cat_array = np.ones((proj_points3.shape[0], 1))
    coordinate = np.concatenate([proj_points3, cat_array], axis=1)
    coordinate3 = (K @ coordinate.T).T[:, 0:3]
    coordinate3 = (coordinate3 / coordinate3[:, 2:3])[:, 0:2]

    mask = (coordinate0[:, 0] >= 0) * (coordinate0[:, 0] < W) * (coordinate0[:, 1] > 0) * (coordinate0[:, 1] < H)
    coordinate0 = np.fliplr(coordinate0[mask].astype(np.int32))
    proj_points0 = proj_points0[mask]

    mask = (coordinate1[:, 0] >= 0) * (coordinate1[:, 0] < W) * (coordinate1[:, 1] > 0) * (coordinate1[:, 1] < H)
    coordinate1 = np.fliplr(coordinate1[mask].astype(np.int32))
    proj_points1 = proj_points1[mask]

    mask = (coordinate2[:, 0] >= 0) * (coordinate2[:, 0] < W) * (coordinate2[:, 1] > 0) * (coordinate2[:, 1] < H)
    coordinate2 = np.fliplr(coordinate2[mask].astype(np.int32))
    proj_points2 = proj_points2[mask]

    mask = (coordinate3[:, 0] >= 0) * (coordinate3[:, 0] < W) * (coordinate3[:, 1] > 0) * (coordinate3[:, 1] < H)
    coordinate3 = np.fliplr(coordinate3[mask].astype(np.int32))
    proj_points3 = proj_points3[mask]

    velodyne_proj_img[coordinate0[:, 0], coordinate0[:, 1]] = proj_points0[:, 2]
    velodyne_proj_img[coordinate1[:, 0], coordinate1[:, 1] + W] = proj_points1[:, 2]
    velodyne_proj_img[coordinate2[:, 0], coordinate2[:, 1] + 2 * W] = proj_points2[:, 2]
    velodyne_proj_img[coordinate3[:, 0], coordinate3[:, 1] + 3 * W] = proj_points3[:, 2]


class SemanticKITTI(Dataset):

    def __init__(self, args, mode='train'):
        self.args = args
        self.data_path = args['dataset_params']['data_path']
        self.H = args['dataset_params']['H']
        self.W = args['dataset_params']['W']
        with open(args['dataset_params']['label_mapping'], 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)

        if mode == 'train':
            split = semkittiyaml['split']['train']
        elif mode == 'val':
            split = semkittiyaml['split']['valid']
        elif mode == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('mode must be Train/val/test')

        self.velodyne_proj = self.read_velodyne_proj(self.args['dataset_params']['velodyne_proj_path'])
        self.velodyne_idx = []
        self.proj_matrix = []
        for i in split:
            self.velodyne_idx += absoluteFilePaths('/'.join([self.data_path, str(i).zfill(2), 'velodyne']),
                                                   num_vote=self.args['dataset_params']['num_vote'])
            calib_path = os.path.join(self.data_path, str(i).zfill(2), "calib.txt")
            calib = self.read_calib(calib_path)
            proj_matrix = np.matmul(calib["P2"], calib["Tr"])
            self.proj_matrix.append(proj_matrix)

    @staticmethod
    def read_velodyne_proj(K_path):
        velodyne_proj = {}
        with open(K_path, 'r') as f:
            data = f.readlines()
            K = data[0].split(':')[1].split(' ')[1:]
            K = np.array(K).reshape(3, 4).astype(np.float32)
            T_velo2img = data[1].split(':')[1].split(' ')[1:]
            T_velo2img = np.array(T_velo2img).reshape(4, 4).astype(np.float32)
            T_4img = data[2].split(':')[1].split(' ')[1:]
            T_4img = np.array(T_4img).reshape(4, 4).astype(np.float32)
            T_xyz = data[3].split(':')[1].split(' ')[1:]
            T_xyz = np.array(T_xyz).reshape(4, 4).astype(np.float32)
            velodyne_proj['K'] = K
            velodyne_proj['T_velo2img'] = T_velo2img
            velodyne_proj['T_4img'] = T_4img
            velodyne_proj['T_xyz'] = T_xyz
        return velodyne_proj

    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    break
                key, value = line.split(':', 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
        calib_out['Tr'] = np.identity(4)  # 4x4 matrix
        calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)

        return calib_out

    def __len__(self):
        return len(self.velodyne_idx)

    def __getitem__(self, index):
        print(index)
        raw_data = np.fromfile(self.velodyne_idx[index], dtype=np.float32).reshape((-1, 4))
        point = raw_data[:, :3]
        distance = np.linalg.norm(point, axis=1).astype(np.float32)
        cat_dim = np.ones((point.shape[0], 1))
        point = np.concatenate([point, cat_dim], axis=1)
        proj_points = (self.velodyne_proj['T_xyz'] @ point.T).T[:, :3]
        proj_distance = np.linalg.norm(proj_points, axis=1).astype(np.float32)
        pitch = np.arcsin(proj_points[:, 2] / proj_distance) / np.pi * 180
        yaw = np.arctan2(-proj_points[:, 1], -proj_points[:, 0]) / np.pi
        proj_x = 0.5 * (yaw + 1)
        jump_index = np.nonzero((yaw[1:] < 0.2) * (yaw[:-1] > 0.8))[0] + 1
        jump = np.zeros_like(proj_x).astype(np.int32)
        jump[jump_index] = 1
        jump = np.cumsum(jump, axis=0)
        denser_coordinate, denser_coordinate_lines = getDenserCoord(pitch, jump, self.velodyne_proj['K'], self.H, self.W,
                                                                    self.args['dataset_params']['scan_point_num'] // 4,
                                                                    self.args['depth_completion_params']['add_lines'])
        sparse_proj_points = []
        for i in range(jump.max()+1):
            mask = np.nonzero(jump == i)
            sparse_proj_points.append(proj_points[mask])

        proj_points = proj_points[deal_occlusion(proj_x, jump)]
        velodyne_proj_img = np.zeros((self.H, 4 * self.W))
        start_rot = -np.pi / 4
        start_rot = [np.cos(start_rot), -np.sin(start_rot), 0, 0, np.sin(start_rot), np.cos(start_rot), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        start_rot = np.array(start_rot).reshape(4, 4).astype(np.float32)
        get_velodyne_proj_img(start_rot, proj_points, velodyne_proj_img, self.H, self.W, self.velodyne_proj)
        velodyne_proj_img = torch.tensor(np.expand_dims(velodyne_proj_img, -1)).permute(2, 0, 1).contiguous().float()
        position = CoordConv.AddCoordsNp(self.H, self.W)
        position = position.call()
        position = totensor(position).float()
        data = {}
        data['raw_data'] = raw_data
        data['K'] = torch.tensor(self.velodyne_proj['K']).cuda()
        data['T_velo2img'] = self.velodyne_proj['T_velo2img']
        data['T_4img'] = self.velodyne_proj['T_4img']
        data['T_xyz'] = self.velodyne_proj['T_xyz']
        data['T_rot'] = start_rot
        data['position'] = position.cuda()
        data['proj_points'] = proj_points
        data['sparse_proj_points'] = sparse_proj_points
        data['proj_distance'] = proj_distance
        data['distance'] = distance
        data['denser_coordinate'] = denser_coordinate
        data['denser_coordinate_lines'] = denser_coordinate_lines
        data['velodyne_proj_img'] = velodyne_proj_img.cuda()
        data['velodyne_proj_img0'] = velodyne_proj_img[:, :, :self.W].cuda()
        data['velodyne_proj_img1'] = velodyne_proj_img[:, :, self.W:2 * self.W].cuda()
        data['velodyne_proj_img2'] = velodyne_proj_img[:, :, 2 * self.W:3 * self.W].cuda()
        data['velodyne_proj_img3'] = velodyne_proj_img[:, :, 3 * self.W:4 * self.W].cuda()
        # data['velodyne_geo_s1'] = velodyne_geo_s1.cuda()
        # data['velodyne_geo_s1_img0'] = velodyne_geo_s1[:, :, :self.W].cuda()
        # data['velodyne_geo_s1_img1'] = velodyne_geo_s1[:, :, self.W:2 * self.W]
        # data['velodyne_geo_s1_img2'] = velodyne_geo_s1[:, :, 2 * self.W:3 * self.W]
        # data['velodyne_geo_s1_img3'] = velodyne_geo_s1[:, :, 3 * self.W:4 * self.W]

        return data
