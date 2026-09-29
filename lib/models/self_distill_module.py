"""
1. Root-joint KD: student 為 root_net 的 root 座標，teacher 為 pose_net root 關節座標，loss 權重由 cfg.KD.ROOT_WEIGHT 控制。
2. 3D Heatmap KD: student 為 root_net 3D heatmap，teacher 由 pose_net joints 合成 3D heatmap，loss 權重由 cfg.KD.HM3D_WEIGHT 控制。
3. 2D Heatmap KD: student 為 backbone 2D heatmap，teacher 由 pose_net joints 投影到影像平面並生成 2D heatmap，loss 權重由 cfg.KD.HM2D_WEIGHT 控制。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.transforms import get_affine_transform, affine_transform
from utils.cameras import project_pose
from dataset.JointsDataset import JointsDataset


class SelfDistillModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.root_id = cfg.DATASET.ROOTIDX
        self.img_size = cfg.NETWORK.IMAGE_SIZE
        self.mse = nn.MSELoss(reduction='none')
        self.joints_dataset = JointsDataset(cfg, image_set='train', is_train=True)

    def root_joint_kd(self, grid_centers, single_pose):
        """L1 between student root centers and teacher (pose_net) root joint.
        grid_centers: [N, 5] (x,y,z, matched_gt_id, conf)
        single_pose:  [N, J, 3] teacher 3D joints (pose_net output)
        """
        t_pose = single_pose.detach()
        valid = (grid_centers[:, 3] >= 0) & (grid_centers[:, 4] >= self.cfg.KD.CONF_THRESH)
        if not valid.any():
            return torch.tensor(0.0, device=grid_centers.device)

        stu_root = grid_centers[valid, :3]
        tea_root = t_pose[valid, self.root_id, :] if isinstance(self.root_id, int) else t_pose[valid, self.root_id, :].mean(dim=1)
        return F.l1_loss(stu_root, tea_root, reduction='mean')

    def heatmap_3d_kd(self, root_cubes, pred):
        """MSE between student 3D heatmap and teacher pseudo 3D heatmap.
        root_cubes: [B, D, H, W] student 3D heatmaps
        pred:       [B, N, J, 5] teacher joints+conf (uses [:,:,:, :3] and [:, :, 0, 4])
        """
        t_pred = pred.detach()
        B = root_cubes.shape[0]
        pseudo_list = []
        for b in range(B):
            cand = (t_pred[b, :, 0, 3] >= 0) & (t_pred[b, :, 0, 4] >= self.cfg.KD.CONF_THRESH)
            if cand.any():
                joints_3d = t_pred[b, cand, :, :3]
                conf = t_pred[b, cand, 0, 4]
                pseudo = self.joints_dataset.generate_3d_target_gpu(joints_3d, conf)
            else:
                pseudo = torch.zeros_like(root_cubes[b], device=root_cubes.device)
            pseudo_list.append(pseudo)
        teacher = torch.stack(pseudo_list, dim=0)  # [B, D, H, W]

        mass = teacher.sum(dim=[1, 2, 3])  # [B]
        valid = mass > 1e-5
        if not valid.any():
            return torch.tensor(0.0, device=root_cubes.device)

        per_voxel = self.mse(root_cubes, teacher)                    # [B, D, H, W]
        num = (per_voxel * teacher).sum(dim=[1, 2, 3])               # [B]
        per_sample = num[valid] / mass[valid].clamp_min(1e-6)        # [#valid]
        return per_sample.mean()

    def heatmap_2d_kd(self, heatmaps, pred, meta):
        """MSE between student 2D heatmaps per view and teacher (projected) 2D heatmaps.
        meta:      list of view dicts; uses meta[c]['center'][i], ['scale'][i], and ['camera']
        pred:      [B, N, J, 5] teacher 3D joints+conf
        heatmaps:  list of length V; each is [B, J, H, W] student 2D heatmaps per view
        """
        device = heatmaps[0].device
        B, N, J, _ = pred.shape
        V = len(heatmaps)
        t_pred = pred.detach()

        total = 0.0
        for i in range(heatmaps[0].shape[0]):  # per sample
            # precompute 2D teachers per view: list of [N, J, H, W]
            tea_per_view = []
            for c in range(V):
                center = meta[c]['center'][i]
                scale = meta[c]['scale'][i]
                trans = torch.as_tensor(get_affine_transform(center, scale, 0, self.img_size), dtype=torch.float, device=device)
                cam = {k: v[i] for k, v in meta[c]['camera'].items()}

                poses3d = t_pred[i, :, :, :3]            # [N, J, 3]
                conf3d = t_pred[i, :, :, 4]              # [N, J]

                # project each person, build [N, J, H, W]
                per_person = []
                for p in range(poses3d.shape[0]):
                    pose2d = project_pose(poses3d[p], cam)                # [J,2]
                    pose2d = torch.stack([affine_transform(pose2d[j], trans) for j in range(J)], dim=0)  # [J,2]
                    pose2d_conf = torch.cat([pose2d, conf3d[p].unsqueeze(-1)], dim=-1)                    # [J,3]
                    hm = self.joints_dataset.generate_input_heatmap_gpu(pose2d_conf.to(device))           # [J,H,W]
                    if isinstance(hm, np.ndarray):
                        hm = torch.from_numpy(hm).to(device)
                    per_person.append(hm)
                tea_per_view.append(torch.stack(per_person, dim=0))  # [N,J,H,W]

            # accumulate loss across views/persons/joints
            loss_i = 0.0
            for c in range(V):
                stu_hm = heatmaps[c][i]        # [J,H,W]
                tea_hm_all = tea_per_view[c]   # [N,J,H,W]
                if 'joints_vis' in meta[c]:
                    joints_vis = meta[c]['joints_vis'][i].to(device)   # [N,J,2]
                else:
                    joints_vis = torch.ones((N, J, 2), device=device)
                for p in range(N):
                    vis_mask = joints_vis[p, :, 0].view(J, 1, 1)
                    for j in range(J):
                        tea = tea_hm_all[p, j]
                        stu = stu_hm[j]
                        conf = (t_pred[i, p, j, 4]).clamp_min(1e-3)
                        loss_i += (self.mse(tea, stu) * vis_mask[j] * conf).mean()
            total += loss_i / V
        return total / heatmaps[0].shape[0]