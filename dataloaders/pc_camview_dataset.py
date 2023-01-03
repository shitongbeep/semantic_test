import os
import numpy as np
import yaml
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torch
from utils import CoordConv
from PIL import Image
from dataloaders.bottomcrop import BottomCrop

totensor = transforms.ToTensor()


def absoluteFilePaths(directory, num_vote):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            for _ in range(num_vote):
                yield os.path.abspath(os.path.join(dirpath, f))


def deal_occlusion(pitch, jump):
    mask = []
    for line_idx in range(jump.max() + 1):
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


def get_velodyne_proj_img(points, velodyne_proj_img, H, W, proj_matrix, trans_matrix, jump, coord_lines):
    xyz1 = points[:]
    pt_jump = jump[:]
    proj_xyz1 = (trans_matrix @ xyz1.T).T[:, 0:3]
    coordinate = (proj_matrix @ xyz1.T).T[:, 0:3]
    mask = coordinate[:, 2] > 0
    coordinate = coordinate[mask]
    xyz1 = xyz1[mask]
    proj_xyz1 = proj_xyz1[mask]
    pt_jump = pt_jump[mask]
    coordinate = (coordinate / coordinate[:, 2:3])[:, 0:2]

    mask = (coordinate[:, 0] >= 0) * (coordinate[:, 0] < W) * (coordinate[:, 1] > 0) * (coordinate[:, 1] < H)
    coordinate = np.fliplr(coordinate[mask].astype(np.int32))
    xyz1 = xyz1[mask]
    proj_xyz1 = proj_xyz1[mask]
    pt_jump = pt_jump[mask]

    velodyne_proj_img[coordinate[:, 0], coordinate[:, 1]] = proj_xyz1[:, 2]

    for line in range(pt_jump.max() + 1):
        cur_line = coordinate[pt_jump == line]
        coord_lines.append(cur_line)


class SemanticKITTI(Dataset):

    def __init__(self, args, mode='train'):
        self.args = args
        self.data_path = args['dataset_params']['data_path']
        self.H = args['dataset_params']['H']
        self.W = args['dataset_params']['W']
        self.inputH = args['dataset_params']['inputH']
        self.inputW = args['dataset_params']['inputW']
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

        self.bottomcrop = BottomCrop([self.H, self.W])

        self.velodyne_idx = []
        self.proj_matrix = {}
        self.proj_matrix_inv = {}
        self.K_list = {}
        self.trans_matrix = {}
        for i in split:
            self.velodyne_idx += absoluteFilePaths('/'.join([self.data_path, str(i).zfill(2), 'velodyne']),
                                                   num_vote=self.args['dataset_params']['num_vote'])
            calib_path = os.path.join(self.data_path, str(i).zfill(2), "calib.txt")
            calib = self.read_calib(calib_path)
            proj_matrix = np.matmul(calib["P2"], calib["Tr"])
            calib['P2'][0, 2] -= (1242 - 1216) / 2
            calib['P2'][1, 2] -= (375 - 352)
            proj_matrix_inv = np.identity(4)
            proj_matrix_inv[:3, :4] = np.matmul(calib["P2"], calib["Tr"])
            self.proj_matrix_inv[i] = (np.linalg.inv(proj_matrix_inv))
            self.K_list[i] = (torch.tensor(calib['P2']))
            self.trans_matrix[i] = (calib["Tr"])
            self.proj_matrix[i] = (proj_matrix)

    @staticmethod
    def read_velodyne_proj(K_path):
        velodyne_proj = {}
        with open(K_path, 'r') as f:
            data = f.readlines()
            K = data[0].split(':')[1].split(' ')[1:]
            K = np.array(K).reshape(3, 4).astype(np.float32)
            T_velo2img = data[1].split(':')[1].split(' ')[1:]
            T_velo2img = np.array(T_velo2img).reshape(4, 4).astype(np.float32)
            velodyne_proj['K'] = K
            velodyne_proj['T_velo2img'] = T_velo2img
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
        # calib_out['P2'][0, 2] -= (1242 - 1216) / 2
        # calib_out['P2'][0, 2] -= (375 - 352) / 2
        calib_out['Tr'] = np.identity(4)  # 4x4 matrix
        calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)

        return calib_out

    def __len__(self):
        return len(self.velodyne_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.velodyne_idx[index], dtype=np.float32).reshape((-1, 4))
        proj_idx = int(self.velodyne_idx[index][-22:-20])
        # print(proj_idx)
        proj_matrix = self.proj_matrix[proj_idx]
        proj_matrix_inv = self.proj_matrix_inv[proj_idx]
        trans_matrix = self.trans_matrix[proj_idx]
        point = raw_data[:, :3]
        cat_dim = np.ones((point.shape[0], 1))
        xyz1 = np.concatenate([point, cat_dim], axis=1)
        distance = np.linalg.norm(point, axis=1).astype(np.float32)
        yaw = np.arctan2(-point[:, 1], -point[:, 0]) / np.pi
        proj_x = 0.5 * (yaw + 1)
        jump_index = np.nonzero((yaw[1:] < 0.2) * (yaw[:-1] > 0.8))[0] + 1
        jump = np.zeros_like(proj_x).astype(np.int32)
        jump[jump_index] = 1
        jump = np.cumsum(jump, axis=0)
        sparse_point = []
        for i in range(jump.max() + 1):
            mask = np.nonzero(jump == i)
            sparse_point.append(point[mask])

        point = point[deal_occlusion(proj_x, jump)]
        coord_lines = []
        velodyne_proj_img = np.zeros((self.inputH, self.inputW))
        get_velodyne_proj_img(xyz1, velodyne_proj_img, self.inputH, self.inputW, proj_matrix, trans_matrix, jump, coord_lines)
        velodyne_proj_img = torch.tensor(np.expand_dims(velodyne_proj_img, -1)).permute(2, 0, 1).contiguous().float()
        position = CoordConv.AddCoordsNp(self.H, self.W)
        position = position.call()
        position = totensor(position).float()

        image_file = self.velodyne_idx[index].replace('velodyne', 'image_2').replace('.bin', '.png')
        image = Image.open(image_file)
        image = self.bottomcrop(image)
        imgarray = np.array(image).astype(np.uint8)
        rgb_img = torch.tensor(imgarray).float().permute(2, 0, 1)
        velodyne_proj_img = self.bottomcrop(velodyne_proj_img)

        data = {}
        data['raw_data'] = raw_data
        data['position'] = position.cuda()
        data['point'] = point
        data['rgb_img'] = rgb_img.cuda()
        data['sparse_point'] = sparse_point
        data['distance'] = distance
        data['K'] = self.K_list[proj_idx].cuda()
        data['velodyne_proj_img'] = velodyne_proj_img.cuda()
        data['proj_matrix'] = proj_matrix
        data['proj_matrix_inv'] = proj_matrix_inv
        data['coord_lines'] = coord_lines

        return data
