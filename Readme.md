# VSDPose

This is the official implementation for:

> **VSDPose: Voxel-based Self-distillation for Multi-view 3D Human Pose Estimation**
> Bo-Han, Chen and Chia-Chi, Tsai
> BMVC 2026 (Poster)

## Installation

* Clone this repo, and we'll call the directory that you cloned VSDPose as ${POSE_ROOT}.
* Install dependencies.

## Data preparation

Datasets:
Please follow the previous work [**VoxelPose**](https://arxiv.org/abs/2004.06239) to prepare the Panoptic, Campus and Shelf datasets.

Backbone :
Following [**VoxelPose**](https://arxiv.org/abs/2004.06239), we use a 2D pose estimator pretrained on the COCO dataset as the backbone. For training the 3D pose estimation model, we use independently sampled 3D human poses from the CMU Panoptic dataset.

Synethsis heatmap:
For the Campus and Shelf datasets, following [**VoxelPose**](https://arxiv.org/abs/2004.06239), we use a 2D pose estimator pretrained on the **COCO** dataset to obtain the 2D poses.

The directory tree should look like this:
${POSE_ROOT}
|-- data
    |-- Shelf
    |   |-- Camera0
    |   |-- ...
    |   |-- Camera4
    |   |-- actorsGT.mat
    |   |-- calibration_shelf.json
    |   |-- pred_shelf_maskrcnn_hrnet_coco.pkl
    |-- CampusSeq1
    |   |-- Camera0
    |   |-- Camera1
    |   |-- Camera2
    |   |-- actorsGT.mat
    |   |-- calibration_campus.json
    |   |-- pred_campus_maskrcnn_hrnet_coco.pkl
    |-- panoptic_training_pose.pkl

## Training

CMU Panoptic dataset

```
python run/train_3d.py --cfg configs/panoptic/resnet50/prn64_cpn80x80x20_960x512_cam5_KD.yaml
```

Shelf/Campus datasets

```
python run/train_3d.py --cfg configs/shelf/prn64_cpn80x80x20.yaml
python run/train_3d.py --cfg configs/campus/prn64_cpn80x80x20.yaml
```

## Evaluation

### Model Weights

The model weights can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1hh8RhO3Y_SdsBbVhsTfR7F7uKzpIwQwL?usp=drive_link).

CMU Panoptic dataset

```
python run/train_3d.py --cfg configs/panoptic/resnet50/prn64_cpn80x80x20_960x512_cam5.yaml
```

Results

|  Threshold (mm)  |    25 |    50 |    75 |   100 |   125 |   150 |
| :--------------: | ----: | ----: | ----: | ----: | ----: | ----: |
|   **AP**   | 92.84 | 98.14 | 98.98 | 99.27 | 99.33 | 99.34 |
| **Recall** | 94.66 | 98.40 | 99.03 | 99.28 | 99.34 | 99.35 |

**MPJPE:** 14.32 mm

Shelf/Campus datasets

```
python run/train_3d.py --cfg configs/shelf/prn64_cpn80x80x20_KD.yaml
python run/train_3d.py --cfg configs/campus/prn64_cpn80x80x20_KD.yaml
```

## Citation

If you use our code or models in your research, please cite our resarch.

## Acknowlegments

The code is based on the VoxelPose (thanks to the authors for sharing their code). Please consider citing their work too.
