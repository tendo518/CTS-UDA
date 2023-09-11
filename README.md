# CTS Sim-to-Real


## Installation

Follow instruction in [OpenPCDet Install Guide](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md)


## Get Started

### Dataset Preparation

Carla3D (Simulation/Source) is available at **not released yet**.

TinySUSCape (already organized in KITTI format) is available from [JST](https://github.com/guangyaooo/JointRCNN).

Please refer to [3D_adapt_auto_driving](https://github.com/cxy1997/3D_adapt_auto_driving#usage) to prepare the KITTI and Lyft dataset. The train/val split txt of KITTI and Lyft used in our experiments is same as the [3D_adapt_auto_driving](https://github.com/cxy1997/3D_adapt_auto_driving#usage).

Build dataset info db with:

```sh
cd tools/
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos cfgs/dataset_configs/{DATASET}.yaml
```

### Training

1. Train a source model on simulation domain (Carla3D):

```sh
cd tools/
python train.py --cfg_file cfgs/carla_models/pointrcnn_carla_org.yaml
```

2. Self-training with target domain dataset:

```sh
python train_st.py 
--cfg_file cfgs/carla_models/pointrcnn_org_to_{DATASET}.yaml \
--mining_at 0 10 30 40 --mining_portion 0.3 0.5 0.7 1.0 \
# for lyft
# --mining_at 0 5 15 25 --mining_portion 0.3 0.5 0.7 1.0 \
--pretrained_model {BEST_MODEL_FROM_SOURCE_TRAIN} \
```
### Testing

```sh
python test.py \ 
--cfg_file cfgs/carla_models/pointrcnn_org_to_{DATASET}.yaml \
--batch_size 8 --workers 8 \
--ckpt {PATH_TO_CHECKPOINT}
```

## Acknowledgement

Our code is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and [JST](https://github.com/guangyaooo/JointRCNN)
