
import torch

def match_grid_centers_by_id(gt_grid_centers, pred_grid_centers):
    """
    Calculate Mean Center Error (MCE) for matched grid centers.
    Args:
        gt_grid_centers: (B, N, 5) torch.Tensor, ground truth grid centers with shape (batch_size, num_person, 5)
                         where 5 = [x, y, z, id, conf]
        pred_grid_centers: (B, N, 5) torch.Tensor, predicted grid centers with shape (batch_size, num_person, 5)
                           where 5 = [x, y, z, id, conf]
    Returns:
        mean_center_error: float, average Euclidean distance between matched gt and pred centers (in mm)
    """
    gt_list = []
    pred_list = []

    B, N, _ = gt_grid_centers.shape

    for b in range(B):
        gt = gt_grid_centers[b]
        pred = pred_grid_centers[b]

        gt_valid = gt[gt[:, 4] > 0]
        pred_valid = pred[pred[:, 4] > 0]

        for gt_entry in gt_valid:
            gt_id = gt_entry[3].item()
            match = pred_valid[pred_valid[:, 3] == gt_id]
            if match.shape[0] > 0:
                gt_list.append(gt_entry[:3])
                pred_list.append(match[0][:3])

    if len(gt_list) == 0:
        return float('nan')

    gt_tensor = torch.stack(gt_list)
    pred_tensor = torch.stack(pred_list)

    center_dist = torch.norm(gt_tensor - pred_tensor, dim=1)
    return center_dist.mean().item()


def calculate_bbox_iou_2d(gt_centers, pred_centers, box_size=2000.0):
    """
    Calculate 2D bbox IoU (in xy plane) for each pair of (GT, Pred) grid centers.
    Args:
        gt_centers: (B, N, 5), each entry is [x, y, z, id, conf]
        pred_centers: (B, N, 5), predicted data
        box_size: bbox side length (in mm, default 2m)
    Returns:
        float: average IoU of all successfully matched bbox (in xy plane)
    """
    iou_list = []
    B, N, _ = gt_centers.shape
    half = box_size / 2

    for b in range(B):
        gt = gt_centers[b]
        pred = pred_centers[b]
        gt_valid = gt[gt[:, 4] > 0]
        pred_valid = pred[pred[:, 4] > 0]

        for gt_entry in gt_valid:
            gt_id = gt_entry[3].item()
            gt_xy = gt_entry[:2]

            match = pred_valid[pred_valid[:, 3] == gt_id]
            if match.shape[0] == 0:
                continue
            pred_xy = match[0][:2]

            gt_min = gt_xy - half
            gt_max = gt_xy + half
            pr_min = pred_xy - half
            pr_max = pred_xy + half

            inter_min = torch.max(gt_min, pr_min)
            inter_max = torch.min(gt_max, pr_max)
            inter = (inter_max - inter_min).clamp(min=0)
            inter_area = inter[0] * inter[1]
            union_area = box_size ** 2 * 2 - inter_area
            iou = inter_area / union_area
            iou_list.append(iou.item())

    if len(iou_list) == 0:
        return float('nan')

    return sum(iou_list) / len(iou_list)
