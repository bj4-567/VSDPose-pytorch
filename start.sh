export OPENCV_LOG_LEVEL=FATAL
# python run/train_3d.py --cfg configs/panoptic/resnet50/prn64_cpn80x80x20_960x512_cam5_KD.yaml
# python run/validate_3d.py --cfg configs/panoptic/resnet50/prn64_cpn80x80x20_960x512_cam5_KD.yaml
python test/evaluate.py --cfg configs/panoptic/resnet50/prn64_cpn80x80x20_960x512_cam5_KD.yaml