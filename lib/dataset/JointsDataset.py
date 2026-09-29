# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import copy
import logging

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform, get_scale

logger = logging.getLogger(__name__)


class JointsDataset(Dataset):

    def __init__(self, cfg, image_set, is_train, transform=None):
        self.cfg = cfg
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.maximum_person = cfg.MULTI_PERSON.MAX_PEOPLE_NUM

        self.is_train = is_train

        this_dir = os.path.dirname(__file__)
        dataset_root = os.path.join(this_dir, '../..', cfg.DATASET.ROOT)
        self.dataset_root = os.path.abspath(dataset_root)
        self.root_id = cfg.DATASET.ROOTIDX
        self.image_set = image_set
        self.dataset_name = cfg.DATASET.TEST_DATASET

        self.data_format = cfg.DATASET.DATA_FORMAT
        self.data_augmentation = cfg.DATASET.DATA_AUGMENTATION

        self.num_views = cfg.DATASET.CAMERA_NUM

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.NETWORK.TARGET_TYPE
        self.image_size = np.array(cfg.NETWORK.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.NETWORK.HEATMAP_SIZE)
        self.sigma = cfg.NETWORK.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.db = []

        self.space_size = np.array(cfg.MULTI_PERSON.SPACE_SIZE)
        self.space_center = np.array(cfg.MULTI_PERSON.SPACE_CENTER)
        self.initial_cube_size = np.array(cfg.MULTI_PERSON.INITIAL_CUBE_SIZE)


    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if data_numpy is None:
            # logger.error('=> fail to read {}'.format(image_file))
            # raise ValueError('Fail to read {}'.format(image_file))
            return None, None, None, None, None, None

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        joints = db_rec['joints_2d']
        joints_vis = db_rec['joints_2d_vis']
        joints_3d = db_rec['joints_3d']
        joints_3d_vis = db_rec['joints_3d_vis']

        nposes = len(joints)
        assert nposes <= self.maximum_person, 'too many persons'

        height, width, _ = data_numpy.shape
        c = np.array([width / 2.0, height / 2.0])
        s = get_scale((width, height), self.image_size)
        r = 0

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans, (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        for n in range(nposes):
            for i in range(len(joints[0])):
                if joints_vis[n][i, 0] > 0.0:
                    joints[n][i, 0:2] = affine_transform(
                        joints[n][i, 0:2], trans)
                    if (np.min(joints[n][i, :2]) < 0 or
                            joints[n][i, 0] >= self.image_size[0] or
                            joints[n][i, 1] >= self.image_size[1]):
                        joints_vis[n][i, :] = 0

        if 'pred_pose2d' in db_rec and db_rec['pred_pose2d'] != None:
            # For convenience, we use predicted poses and corresponding values at the original heatmaps
            # to generate 2d heatmaps for Campus and Shelf dataset.
            # You can also use other 2d backbone trained on COCO to generate 2d heatmaps directly.
            pred_pose2d = db_rec['pred_pose2d']
            for n in range(len(pred_pose2d)):
                for i in range(len(pred_pose2d[n])):
                    pred_pose2d[n][i, 0:2] = affine_transform(pred_pose2d[n][i, 0:2], trans)

            input_heatmap = self.generate_input_heatmap(pred_pose2d)
            input_heatmap = torch.from_numpy(input_heatmap)
        else:
            input_heatmap = torch.zeros(self.cfg.NETWORK.NUM_JOINTS, self.heatmap_size[1], self.heatmap_size[0])

        target_heatmap, target_weight = self.generate_target_heatmap(
            joints, joints_vis)
        target_heatmap = torch.from_numpy(target_heatmap)
        target_weight = torch.from_numpy(target_weight)

        # make joints and joints_vis having same shape
        joints_u = np.zeros((self.maximum_person, self.num_joints, 2))
        joints_vis_u = np.zeros((self.maximum_person, self.num_joints, 2))
        for i in range(nposes):
            joints_u[i] = joints[i]
            joints_vis_u[i] = joints_vis[i]

        joints_3d_u = np.zeros((self.maximum_person, self.num_joints, 3))
        joints_3d_vis_u = np.zeros((self.maximum_person, self.num_joints, 3))
        for i in range(nposes):
            joints_3d_u[i] = joints_3d[i][:, 0:3]
            joints_3d_vis_u[i] = joints_3d_vis[i][:, 0:3]

        target_3d = self.generate_3d_target_gpu(joints_3d)
        # target_3d = torch.from_numpy(target_3d)

        if isinstance(self.root_id, int):
            roots_3d = joints_3d_u[:, self.root_id]
        elif isinstance(self.root_id, list):
            roots_3d = np.mean([joints_3d_u[:, j] for j in self.root_id], axis=0)
        meta = {
            'image': image_file,
            'num_person': nposes,
            'joints_3d': joints_3d_u,
            'joints_3d_vis': joints_3d_vis_u,
            'roots_3d': roots_3d,
            'joints': joints_u,
            'joints_vis': joints_vis_u,
            'center': c,
            'scale': s,
            'rotation': r,
            'camera': db_rec['camera']
        }

        return input, target_heatmap, target_weight, target_3d, meta, input_heatmap

    def compute_human_scale(self, pose, joints_vis):
        idx = joints_vis[:, 0] == 1
        if np.sum(idx) == 0:
            return 0
        minx, maxx = np.min(pose[idx, 0]), np.max(pose[idx, 0])
        miny, maxy = np.min(pose[idx, 1]), np.max(pose[idx, 1])
        # return np.clip((maxy - miny) * (maxx - minx), 1.0 / 4 * 256**2,
        #                4 * 256**2)
        return np.clip(np.maximum(maxy - miny, maxx - minx)**2,  1.0 / 4 * 96**2, 4 * 96**2)

    def generate_target_heatmap(self, joints, joints_vis):
        '''
        :param joints:  [[num_joints, 3]]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        nposes = len(joints)
        num_joints = self.num_joints
        target_weight = np.zeros((num_joints, 1), dtype=np.float32)
        for i in range(num_joints):
            for n in range(nposes):
                if joints_vis[n][i, 0] == 1:
                    target_weight[i, 0] = 1

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros(
                (num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                dtype=np.float32)
            feat_stride = self.image_size / self.heatmap_size

            for n in range(nposes):
                human_scale = 2 * self.compute_human_scale(joints[n] / feat_stride, joints_vis[n])
                if human_scale == 0:
                    continue

                cur_sigma = self.sigma * np.sqrt((human_scale / (96.0 * 96.0)))
                tmp_size = cur_sigma * 3
                for joint_id in range(num_joints):
                    feat_stride = self.image_size / self.heatmap_size
                    mu_x = int(joints[n][joint_id][0] / feat_stride[0])
                    mu_y = int(joints[n][joint_id][1] / feat_stride[1])
                    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                    if joints_vis[n][joint_id, 0] == 0 or \
                            ul[0] >= self.heatmap_size[0] or \
                            ul[1] >= self.heatmap_size[1] \
                            or br[0] < 0 or br[1] < 0:
                        continue

                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, np.newaxis]
                    x0 = y0 = size // 2
                    g = np.exp(
                        -((x - x0)**2 + (y - y0)**2) / (2 * cur_sigma**2))

                    # Usable gaussian range
                    g_x = max(0,
                              -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                    g_y = max(0,
                              -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                    img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]],
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
                target = np.clip(target, 0, 1)

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def generate_3d_target(self, joints_3d):
        num_people = len(joints_3d)

        space_size = self.space_size
        space_center = self.space_center
        cube_size = self.initial_cube_size
        grid1Dx = np.linspace(-space_size[0] / 2, space_size[0] / 2, cube_size[0]) + space_center[0]
        grid1Dy = np.linspace(-space_size[1] / 2, space_size[1] / 2, cube_size[1]) + space_center[1]
        grid1Dz = np.linspace(-space_size[2] / 2, space_size[2] / 2, cube_size[2]) + space_center[2]

        target = np.zeros((cube_size[0], cube_size[1], cube_size[2]), dtype=np.float32)
        cur_sigma = 200.0

        for n in range(num_people):
            joint_id = self.root_id  # mid-hip
            if isinstance(joint_id, int):
                mu_x = joints_3d[n][joint_id][0]
                mu_y = joints_3d[n][joint_id][1]
                mu_z = joints_3d[n][joint_id][2]
            elif isinstance(joint_id, list):
                mu_x = (joints_3d[n][joint_id[0]][0] + joints_3d[n][joint_id[1]][0]) / 2.0
                mu_y = (joints_3d[n][joint_id[0]][1] + joints_3d[n][joint_id[1]][1]) / 2.0
                mu_z = (joints_3d[n][joint_id[0]][2] + joints_3d[n][joint_id[1]][2]) / 2.0
            i_x = [np.searchsorted(grid1Dx,  mu_x - 3 * cur_sigma),
                       np.searchsorted(grid1Dx,  mu_x + 3 * cur_sigma, 'right')]
            i_y = [np.searchsorted(grid1Dy,  mu_y - 3 * cur_sigma),
                       np.searchsorted(grid1Dy,  mu_y + 3 * cur_sigma, 'right')]
            i_z = [np.searchsorted(grid1Dz,  mu_z - 3 * cur_sigma),
                       np.searchsorted(grid1Dz,  mu_z + 3 * cur_sigma, 'right')]
            if i_x[0] >= i_x[1] or i_y[0] >= i_y[1] or i_z[0] >= i_z[1]:
                continue

            gridx, gridy, gridz = np.meshgrid(grid1Dx[i_x[0]:i_x[1]], grid1Dy[i_y[0]:i_y[1]], grid1Dz[i_z[0]:i_z[1]], indexing='ij')
            g = np.exp(-((gridx - mu_x) ** 2 + (gridy - mu_y) ** 2 + (gridz - mu_z) ** 2) / (2 * cur_sigma ** 2))
            target[i_x[0]:i_x[1], i_y[0]:i_y[1], i_z[0]:i_z[1]] = np.maximum(target[i_x[0]:i_x[1], i_y[0]:i_y[1], i_z[0]:i_z[1]], g)

        target = np.clip(target, 0, 1)
        return target

    def generate_3d_target_gpu(self, joints_3d, confidence_weight=None):
        if isinstance(joints_3d, list):
            if len(joints_3d) == 0:
                device = torch.device('cpu')
                cube_size = self.initial_cube_size
                return torch.zeros((cube_size[0], cube_size[1], cube_size[2]), dtype=torch.float32, device=device)
            joints_3d = [torch.from_numpy(j) if isinstance(j, np.ndarray) else j for j in joints_3d]
            joints_3d = torch.stack(joints_3d)

        if joints_3d is None or (hasattr(joints_3d, 'shape') and joints_3d.shape[0] == 0) or (isinstance(joints_3d, list) and len(joints_3d) == 0):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return torch.zeros((cube_size[0], cube_size[1], cube_size[2]), dtype=torch.float32, device=device)

        if isinstance(joints_3d, np.ndarray):
            joints_3d = torch.from_numpy(joints_3d)
        joints_3d = joints_3d.contiguous().to(dtype=torch.float32)
        device = joints_3d.device

        num_people = joints_3d.shape[0]

        space_size = torch.tensor(self.space_size, dtype=torch.float32)
        space_center = torch.tensor(self.space_center, dtype=torch.float32)
        cube_size = self.initial_cube_size
        grid1Dx = torch.linspace(-space_size[0] / 2, space_size[0] / 2, cube_size[0], device=device) + space_center[0]
        grid1Dy = torch.linspace(-space_size[1] / 2, space_size[1] / 2, cube_size[1], device=device) + space_center[1]
        grid1Dz = torch.linspace(-space_size[2] / 2, space_size[2] / 2, cube_size[2], device=device) + space_center[2]

        target = torch.zeros((cube_size[0], cube_size[1], cube_size[2]), dtype=torch.float32, device=device)
        cur_sigma = 200.0

        for n in range(num_people):
            joint_id = self.root_id  # mid-hip
            if isinstance(joint_id, int):
                mu = joints_3d[n, joint_id]
            elif isinstance(joint_id, list):
                mu = (joints_3d[n, joint_id[0]] + joints_3d[n, joint_id[1]]) / 2
            else:
                continue

            mu_x, mu_y, mu_z = mu[0], mu[1], mu[2]

            x_start = torch.searchsorted(grid1Dx, mu_x - 3 * cur_sigma)
            x_end = torch.searchsorted(grid1Dx, mu_x + 3 * cur_sigma, right=True)
            y_start = torch.searchsorted(grid1Dy, mu_y - 3 * cur_sigma)
            y_end = torch.searchsorted(grid1Dy, mu_y + 3 * cur_sigma, right=True)
            z_start = torch.searchsorted(grid1Dz, mu_z - 3 * cur_sigma)
            z_end = torch.searchsorted(grid1Dz, mu_z + 3 * cur_sigma, right=True)

            if x_end <= x_start or y_end <= y_start or z_end <= z_start:
                continue

            gridx, gridy, gridz = torch.meshgrid(grid1Dx[x_start:x_end], grid1Dy[y_start:y_end], grid1Dz[z_start:z_end], indexing='ij')
            g = torch.exp(-((gridx - mu_x) ** 2 + (gridy - mu_y) ** 2 + (gridz - mu_z) ** 2) / (2 * cur_sigma ** 2))

            if confidence_weight is not None:
                g *= confidence_weight[n]

            target[x_start:x_end, y_start:y_end, z_start:z_end] = torch.maximum(target[x_start:x_end, y_start:y_end, z_start:z_end], g)

        target = target.clamp(0, 1)
        return target


    def generate_input_heatmap(self, joints):
        '''
        :param joints:  [[num_joints, 3]]
        :param joints_vis: [num_joints, 3]
        :return: input_heatmap
        '''
        nposes = len(joints)
        num_joints = self.cfg.NETWORK.NUM_JOINTS

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros(
                (num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                dtype=np.float32)
            feat_stride = self.image_size / self.heatmap_size

            for n in range(nposes):
                human_scale = 2 * self.compute_human_scale(joints[n][:, 0:2] / feat_stride, np.ones((num_joints, 1)))
                if human_scale == 0:
                    continue

                cur_sigma = self.sigma * np.sqrt((human_scale / (96.0 * 96.0)))
                tmp_size = cur_sigma * 3
                for joint_id in range(num_joints):
                    feat_stride = self.image_size / self.heatmap_size
                    mu_x = int(joints[n][joint_id][0] / feat_stride[0])
                    mu_y = int(joints[n][joint_id][1] / feat_stride[1])
                    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                    if ul[0] >= self.heatmap_size[0] or \
                            ul[1] >= self.heatmap_size[1] \
                            or br[0] < 0 or br[1] < 0:
                        continue

                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, np.newaxis]
                    x0 = y0 = size // 2
                    if 'campus' in self.dataset_name:
                        max_value = 1.0
                    else:
                        max_value = joints[n][joint_id][2] if len(joints[n][joint_id]) == 3 else 1.0
                        # max_value = max_value**0.5
                    g = np.exp(
                        -((x - x0)**2 + (y - y0)**2) / (2 * cur_sigma**2)) * max_value

                    # Usable gaussian range
                    g_x = max(0,
                              -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                    g_y = max(0,
                              -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                    img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]],
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
                target = np.clip(target, 0, 1)

        return target


    def generate_input_heatmap_gpu(self, joints, device='cuda'):
        '''
        :param joints: torch.Tensor [nposes, num_joints, 3] (x, y, score)
        :return: heatmap [num_joints, H, W] on GPU
        '''
        nposes = joints.shape[0]
        num_joints = self.cfg.NETWORK.NUM_JOINTS
        joints = joints.to(device)

        assert self.target_type == 'gaussian', 'Only support gaussian map now!'

        target = torch.zeros((num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                             dtype=torch.float32, device=device)

        feat_stride = torch.tensor(self.image_size, dtype=torch.float32, device=device) / \
                      torch.tensor(self.heatmap_size, dtype=torch.float32, device=device)

        for n in range(nposes):
            # [num_joints, 2]
            joints_xy = joints[n][:, :2] / feat_stride
            joints_conf = joints[n][:, 2] if joints.shape[2] == 3 else torch.ones(num_joints, device=device)

            human_scale = 2 * self.compute_human_scale(joints_xy.cpu().numpy(), np.ones((num_joints, 1)))
            if human_scale == 0:
                continue

            cur_sigma = self.sigma * torch.sqrt(torch.tensor(human_scale / (96.0 * 96.0), dtype=torch.float32, device=device))
            tmp_size = cur_sigma * 3
            size = int(2 * tmp_size + 1)

            # Prepare Gaussian patch
            x = torch.arange(0, size, dtype=torch.float32, device=device)
            y = x[:, None]
            x0 = y0 = size // 2
            base_g = torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * cur_sigma ** 2))

            for joint_id in range(num_joints):
                mu_x = int(joints[n, joint_id, 0] / feat_stride[0])
                mu_y = int(joints[n, joint_id, 1] / feat_stride[1])
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] or \
                   br[0] < 0 or br[1] < 0:
                    continue

                # Confidence
                if 'campus' in self.dataset_name:
                    max_value = 1.0
                else:
                    max_value = joints_conf[joint_id]

                g = base_g * max_value

                # 使用與 CPU 版本一致的 slicing 保證形狀相同
                g_x0 = max(0, -ul[0])
                g_x1 = min(br[0], self.heatmap_size[0]) - ul[0]
                g_y0 = max(0, -ul[1])
                g_y1 = min(br[1], self.heatmap_size[1]) - ul[1]

                img_x0 = max(0, ul[0])
                img_x1 = min(br[0], self.heatmap_size[0])
                img_y0 = max(0, ul[1])
                img_y1 = min(br[1], self.heatmap_size[1])

                if img_y1 <= img_y0 or img_x1 <= img_x0:
                    continue

                g_crop = g[g_y0:g_y1, g_x0:g_x1]
                tgt_crop = target[joint_id, img_y0:img_y1, img_x0:img_x1]

                if g_crop.shape != tgt_crop.shape:
                    continue  # 安全保險

                target[joint_id, img_y0:img_y1, img_x0:img_x1] = torch.maximum(
                    tgt_crop, g_crop
                )

        return torch.clamp(target, 0, 1)


