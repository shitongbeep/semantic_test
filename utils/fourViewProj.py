import numpy as np
from matplotlib import cm
from PIL import Image


class FourViewProj():

    def __init__(self, H, W, start_rot=-np.pi / 4):
        self.H = H
        self.W = W
        start_rot = [np.cos(start_rot), -np.sin(start_rot), 0, 0, np.sin(start_rot), np.cos(start_rot), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        self.start_rot = np.array(start_rot).reshape(4, 4).astype(np.float32)
        cmap = cm.get_cmap('viridis')
        sm = cm.ScalarMappable(cmap=cmap)
        self.color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 0:3]

    def forward(self, pc_path, calib_path):
        with open(calib_path, 'r') as f:
            data = f.readlines()
            K = data[0].split(':')[1].split(' ')[1:]
            self.K = np.array(K).reshape(3, 4).astype(np.float32)
            T_velo2img = data[1].split(':')[1].split(' ')[1:]
            self.T_velo2img = np.array(T_velo2img).reshape(4, 4).astype(np.float32)
            T_4img = data[2].split(':')[1].split(' ')[1:]
            self.T_4img = np.array(T_4img).reshape(4, 4).astype(np.float32)
            T_xyz = data[3].split(':')[1].split(' ')[1:]
            self.T_xyz = np.array(T_xyz).reshape(4, 4).astype(np.float32)

        pc_data = np.fromfile(pc_path, np.float32).reshape((-1, 4))
        xyz = pc_data[:, :3]
        xyz1 = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
        proj_xyz = (self.T_xyz[:3, :] @ xyz1.T).T
        pc_range = np.linalg.norm(xyz, axis=1)
        pitch = np.arcsin(proj_xyz[:, 2] / pc_range) / np.pi * 180
        yaw = np.arctan2(-proj_xyz[:, 1], -proj_xyz[:, 0]) / np.pi
        yaw = 0.5 * (yaw + 1)
        jump_idx = ((yaw[1:] < 0.2) * (yaw[:-1] > 0.8))
        jump_idx = np.nonzero(jump_idx)[0] + 1
        jump = np.zeros_like(pitch).astype(np.uint32)
        jump[jump_idx] = 1
        jump = np.cumsum(jump).astype(np.uint32)

        self.velodyne_proj_img = np.zeros((self.H, 4 * self.W))
        self.get_velodyne_proj_img(proj_xyz, self.velodyne_proj_img)
        # self.velodyne_proj_img0 = self.velodyne_proj_img[:, self.W * 0:self.W * 1, ...]
        # self.velodyne_proj_img1 = self.velodyne_proj_img[:, self.W * 1:self.W * 2, ...]
        # self.velodyne_proj_img2 = self.velodyne_proj_img[:, self.W * 2:self.W * 3, ...]
        # self.velodyne_proj_img3 = self.velodyne_proj_img[:, self.W * 3:self.W * 4, ...]

    def get_velodyne_proj_img(self, proj_points, velodyne_proj_img):
        K = self.K
        T_4img = self.T_4img
        T_velo2img = self.T_velo2img
        H = self.H
        W = self.W
        proj_points0 = (self.start_rot[:3, :3] @ proj_points.T).T
        mask = proj_points0[:, 0] > 0
        proj_points0 = proj_points0[mask]
        proj_points0 = (T_velo2img[:3, :3] @ proj_points0.T).T
        # proj_distance0 = proj_distance[mask]
        cat_array = np.ones((proj_points0.shape[0], 1))
        coordinate = np.concatenate([proj_points0, cat_array], axis=1)
        coordinate0 = (K @ coordinate.T).T[:, 0:3]
        coordinate0 = (coordinate0 / coordinate0[:, 2:3])[:, 0:2]

        proj_points1 = (self.start_rot[:3, :3] @ proj_points.T).T
        mask = proj_points1[:, 1] > 0
        proj_points1 = proj_points1[mask]
        proj_points1 = (T_velo2img[:3, :3] @ T_4img[:3, :3] @ proj_points1.T).T
        # proj_distance1 = proj_distance[mask]
        cat_array = np.ones((proj_points1.shape[0], 1))
        coordinate = np.concatenate([proj_points1, cat_array], axis=1)
        coordinate1 = (K @ coordinate.T).T[:, 0:3]
        coordinate1 = (coordinate1 / coordinate1[:, 2:3])[:, 0:2]

        proj_points2 = (self.start_rot[:3, :3] @ proj_points.T).T
        mask = proj_points2[:, 0] < 0
        proj_points2 = proj_points2[mask]
        proj_points2 = (T_velo2img[:3, :3] @ T_4img[:3, :3] @ T_4img[:3, :3] @ proj_points2.T).T
        # proj_distance2 = proj_distance[mask]
        cat_array = np.ones((proj_points2.shape[0], 1))
        coordinate = np.concatenate([proj_points2, cat_array], axis=1)
        coordinate2 = (K @ coordinate.T).T[:, 0:3]
        coordinate2 = (coordinate2 / coordinate2[:, 2:3])[:, 0:2]

        proj_points3 = (self.start_rot[:3, :3] @ proj_points.T).T
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

    def show4ProjImg(self):
        for i in range(4):
            velodyne_proj_img0 = self.velodyne_proj_img[:, self.W * i:self.W * (i + 1), ...]
            depth_img0 = (velodyne_proj_img0 / velodyne_proj_img0.max() * 255.).astype(np.uint8)
            depth_rgb0 = self.color_range[depth_img0].astype(np.uint8)
            # depth_img0[depth_img0 > 0] = 255
            # depth_rgb0 = np.concatenate([depth_rgb0, np.expand_dims(depth_img0, -1)], axis=-1)
            show_img = Image.fromarray(depth_rgb0)
            show_img.show()

    def showProjImg(self):
        velodyne_proj_img0 = self.velodyne_proj_img
        depth_img0 = (velodyne_proj_img0 / velodyne_proj_img0.max() * 255.).astype(np.uint8)
        depth_rgb0 = self.color_range[depth_img0].astype(np.uint8)
        # depth_img0[depth_img0 > 0] = 255
        # depth_rgb0 = np.concatenate([depth_rgb0, np.expand_dims(depth_img0, -1)], axis=-1)
        show_img = Image.fromarray(depth_rgb0)
        show_img.show()

    def compareProjImg(self, another_pc):
        compare_velodyne_proj_img = np.zeros((2 * self.H, 4 * self.W))
        compare_velodyne_proj_img[:self.H, :] = self.velodyne_proj_img
        compare_velodyne_proj_img[self.H:2 * self.H, :] = another_pc.velodyne_proj_img
        depth_img0 = (compare_velodyne_proj_img / compare_velodyne_proj_img.max() * 255.).astype(np.uint8)
        depth_rgb0 = self.color_range[depth_img0].astype(np.uint8)
        # depth_img0[depth_img0 > 0] = 255
        # depth_rgb0 = np.concatenate([depth_rgb0, np.expand_dims(depth_img0, -1)], axis=-1)
        show_img = Image.fromarray(depth_rgb0)
        show_img.show()

    def compare4ProjImg(self, another_pc):
        compare_velodyne_proj_img = np.zeros((2 * self.H, 4 * self.W))
        compare_velodyne_proj_img[:self.H, :] = self.velodyne_proj_img
        compare_velodyne_proj_img[self.H:2 * self.H, :] = another_pc.velodyne_proj_img
        for i in range(4):
            velodyne_proj_img0 = compare_velodyne_proj_img[:, self.W * i:self.W * (i + 1), ...]
            depth_img0 = (velodyne_proj_img0 / velodyne_proj_img0.max() * 255.).astype(np.uint8)
            depth_rgb0 = self.color_range[depth_img0].astype(np.uint8)
            # depth_img0[depth_img0 > 0] = 255
            # depth_rgb0 = np.concatenate([depth_rgb0, np.expand_dims(depth_img0, -1)], axis=-1)
            show_img = Image.fromarray(depth_rgb0)
            show_img.show()
