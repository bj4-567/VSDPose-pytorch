# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from models import pose_resnet
from models.cuboid_proposal_net import CuboidProposalNet
from models.pose_regression_net import PoseRegressionNet
from core.loss import PerJointMSELoss
from core.loss import PerJointL1Loss
from models.self_distill_module import SelfDistillModule


class MultiPersonPoseNet(nn.Module):
    def __init__(self, backbone, cfg):
        super(MultiPersonPoseNet, self).__init__()
        self.cfg = cfg
        self.num_cand = cfg.MULTI_PERSON.MAX_PEOPLE_NUM
        self.num_joints = cfg.NETWORK.NUM_JOINTS

        self.backbone = backbone
        self.root_net = CuboidProposalNet(cfg)
        self.pose_net = PoseRegressionNet(cfg)
        self.kd_module = SelfDistillModule(cfg)

        self.USE_GT = cfg.NETWORK.USE_GT
        self.root_id = cfg.DATASET.ROOTIDX
        self.dataset_name = cfg.DATASET.TEST_DATASET
        self.kd_start_epoch = cfg.KD.START_EPOCH

    def forward(self, views=None, meta=None, targets_2d=None, weights_2d=None, targets_3d=None, input_heatmaps=None):
        if views is not None:
            all_heatmaps = []
            for view in views:
                heatmaps = self.backbone(view)
                all_heatmaps.append(heatmaps)
        else:
            all_heatmaps = input_heatmaps

        # all_heatmaps = targets_2d
        device = all_heatmaps[0].device
        batch_size = all_heatmaps[0].shape[0]

        current_epoch = meta[0].get('epoch', 0)
        warmup_factor = min(1.0, float(current_epoch) / float(self.kd_start_epoch + 1e-5))  # avoid div0
        use_kd = self.training and current_epoch >= self.kd_start_epoch

        # initialize
        root_cubes = None
        loss_2d = torch.tensor(0.0, device=device)
        loss_3d = torch.tensor(0.0, device=device)
        loss_cord = torch.tensor(0.0, device=device)
        loss_kd_root = torch.tensor(0.0, device=device)
        loss_kd_hm3d = torch.tensor(0.0, device=device)
        loss_kd_hm2d = torch.tensor(0.0, device=device)
        w_kd_root = self.cfg.KD.ROOT_WEIGHT
        w_kd_hm3d = self.cfg.KD.HM3D_WEIGHT
        w_kd_hm2d = self.cfg.KD.HM2D_WEIGHT

        # calculate 2D heatmap loss
        criterion = PerJointMSELoss().to(device)
        if self.cfg.LOSS.USE_2D_LOSS and targets_2d is not None:        # defalut: LOSS.USE_2D_LOSS = False
            for t, w, o in zip(targets_2d, weights_2d, all_heatmaps):
                loss_2d += criterion(o, t, True, w)
            loss_2d /= len(all_heatmaps)

        if self.USE_GT:
            num_person = meta[0]['num_person']
            grid_centers = torch.zeros(batch_size, self.num_cand, 5, device=device)
            grid_centers[:, :, 0:3] = meta[0]['roots_3d'].float()
            grid_centers[:, :, 3] = -1.0
            for i in range(batch_size):
                grid_centers[i, :num_person[i], 3] = torch.tensor(range(num_person[i]), device=device)
                grid_centers[i, :num_person[i], 4] = 1.0
        else:
            root_cubes, grid_centers = self.root_net(all_heatmaps, meta)

            # calculate 3D heatmap loss
            if targets_3d is not None:
                loss_3d = criterion(root_cubes, targets_3d)


        pred = torch.zeros(batch_size, self.num_cand, self.num_joints, 5, device=device)
        pred[:, :, :, 3:] = grid_centers[:, :, 3:].reshape(batch_size, -1, 1, 2)  # matched gt

        criterion_cord = PerJointL1Loss().to(device)
        count = 0

        for n in range(self.num_cand):
            index = (pred[:, n, 0, 3] >= 0)
            if torch.sum(index) > 0:
                single_pose = self.pose_net(all_heatmaps, meta, grid_centers[:, n])
                pred[:, n, :, 0:3] = single_pose.detach()

                # calculate 3D pose loss
                if self.training and 'joints_3d' in meta[0] and 'joints_3d_vis' in meta[0]:
                    gt_3d = meta[0]['joints_3d'].float()
                    for i in range(batch_size):
                        if pred[i, n, 0, 3] >= 0:
                            targets = gt_3d[i:i + 1, pred[i, n, 0, 3].long()]
                            weights_3d = meta[0]['joints_3d_vis'][i:i + 1, pred[i, n, 0, 3].long(), :, 0:1].float()
                            count += 1
                            loss_cord = (loss_cord * (count - 1) +
                                         criterion_cord(single_pose[i:i + 1], targets, True, weights_3d)) / count

                # calculate root joint distillation loss
                if self.training and use_kd and self.cfg.KD.ROOT:
                    current_root_loss = self.kd_module.root_joint_kd(grid_centers[:, n], single_pose)
                    loss_kd_root += current_root_loss

                del single_pose

        # calculate 3D heatmap distillation loss
        if self.training and use_kd and self.cfg.KD.HM3D and root_cubes is not None:
            loss_kd_hm3d = self.kd_module.heatmap_3d_kd(root_cubes, pred)

        # calculate 2D heatmap distillation loss
        if self.training and use_kd and self.cfg.KD.HM2D:
            loss_kd_hm2d = self.kd_module.heatmap_2d_kd(all_heatmaps, pred, meta)

        del root_cubes

        loss_dict = {
            'loss_2d': loss_2d,
            'loss_3d': loss_3d,
            'loss_cord': loss_cord,
            'loss_kd_root': w_kd_root * loss_kd_root,
            'loss_kd_hm3d': w_kd_hm3d * loss_kd_hm3d,
            'loss_kd_hm2d': w_kd_hm2d * loss_kd_hm2d,
            'total': loss_2d + loss_3d + loss_cord +
                        w_kd_root * loss_kd_root +
                        w_kd_hm3d * loss_kd_hm3d +
                        w_kd_hm2d * loss_kd_hm2d
        }

        return pred, all_heatmaps, grid_centers, loss_dict


def get_multi_person_pose_net(cfg, is_train=True):
    if cfg.BACKBONE_MODEL:
        backbone = eval(cfg.BACKBONE_MODEL + '.get_pose_net')(cfg, is_train=is_train)
    else:
        backbone = None
    model = MultiPersonPoseNet(backbone, cfg)
    return model
