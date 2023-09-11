from functools import partial

import numpy as np
import torch

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from . import augmentor_utils, database_sampler


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = []
        aug_config_list = (
            augmentor_configs
            if isinstance(augmentor_configs, list)
            else augmentor_configs.AUG_CONFIG_LIST
        )

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger,
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d["logger"]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict["gt_boxes"], data_dict["points"]
        for cur_axis in config["ALONG_AXIS_LIST"]:
            assert cur_axis in ["x", "y"]
            gt_boxes, points = getattr(
                augmentor_utils, "random_flip_along_%s" % cur_axis
            )(
                gt_boxes,
                points,
            )

        data_dict["gt_boxes"] = gt_boxes
        data_dict["points"] = points
        return data_dict

    # def random_jitter(self, data_dict=None, config=None):
    #     if data_dict is None:
    #         return partial(self.random_jitter, config=config)
    #     gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
    #     for cur_axis in config['ALONG_AXIS_LIST']:
    #         assert cur_axis in ['x', 'y']
    #         gt_boxes, points = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
    #             gt_boxes, points,
    #         )
    #
    #     data_dict['gt_boxes'] = gt_boxes
    #     data_dict['points'] = points
    #     return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config["WORLD_ROT_ANGLE"]
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.global_rotation(
            data_dict["gt_boxes"], data_dict["points"], rot_range=rot_range
        )

        data_dict["gt_boxes"] = gt_boxes
        data_dict["points"] = points
        return data_dict

    def random_object_jitter(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_object_jitter, config=config)

        gt_boxes, points = data_dict["gt_boxes"], data_dict["points"]
        if gt_boxes.shape[1] > 7:
            gt_boxes = gt_boxes[:, :7]

        num_obj = len(gt_boxes)
        if num_obj == 0:
            return data_dict

        noise_rotation = np.random.uniform(config["ROT"][0], config["ROT"][1], num_obj)
        noise_scale = np.random.uniform(config["SCALE"][0], config["SCALE"][1], num_obj)
        noise_transition = np.random.uniform(
            config["TRANSITION"][0], config["TRANSITION"][1], (num_obj, 2)
        )
        noise_whl_ratio = np.random.uniform(
            config["WHL_RATIO"][0], config["WHL_RATIO"][1], (num_obj, 3)
        )

        # get masks of points inside boxes
        point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(
            torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
        ).numpy()

        obj_points_list = []
        gt_boxes_size = gt_boxes[:, 3:6]
        # scale the objects
        for i in range(num_obj):
            obj_mask = point_masks[i] > 0
            h = gt_boxes[i, 5]
            obj_points = points[obj_mask]  # get object points within the gt box
            obj_points[:, :3] -= gt_boxes[i, :3]  # relative to box center

            # scale
            scale = noise_scale[i] * noise_whl_ratio[i]
            obj_points[:, :3] *= scale
            gt_boxes[i, 3:6] *= scale

            # rotation
            obj_points[:, :3] = common_utils.rotate_points_along_z(
                obj_points[None, :, :3], noise_rotation[i : i + 1]
            )[0]
            gt_boxes[i, 6] += noise_rotation[i]

            # transition
            gt_boxes[i, :2] += noise_transition[i]
            dz = (gt_boxes[i, 5] - h) / 2
            gt_boxes[i, 2] += dz
            obj_points[:, :3] += gt_boxes[i, :3]  # back to global coordinate

            points[obj_mask] = obj_points

            obj_points_list.append(obj_points)

        # # remove points inside boxes
        # points = points[point_masks.sum(0) == 0]
        #
        # # merge points
        # obj_points = np.concatenate(obj_points_list, axis=0)
        # points = np.concatenate([points, obj_points], axis=0)

        data_dict["points"] = points
        data_dict["gt_boxes"][:, :7] = gt_boxes
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        gt_boxes, points = augmentor_utils.global_scaling(
            data_dict["gt_boxes"], data_dict["points"], config["WORLD_SCALE_RANGE"]
        )
        data_dict["gt_boxes"] = gt_boxes
        data_dict["points"] = points
        return data_dict

    # obtained from https://github.com/CVMI-Lab/ST3D/blob/2634561684dfbafce6ed86e50c6f70f51988593c/pcdet/datasets/augmentor/data_augmentor.py#L59C1-L70C25
    def random_object_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_object_scaling, config=config)
        points, gt_boxes = augmentor_utils.scale_pre_object(
            data_dict['gt_boxes'], data_dict['points'],
            gt_boxes_mask=data_dict['gt_boxes_mask'],
            scale_perturb=config['SCALE_UNIFORM_NOISE']
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        data_dict["gt_boxes"][:, 6] = common_utils.limit_period(
            data_dict["gt_boxes"][:, 6], offset=0.5, period=2 * np.pi
        )
        if "calib" in data_dict:
            data_dict.pop("calib")
        if "road_plane" in data_dict:
            data_dict.pop("road_plane")
        if "gt_boxes_mask" in data_dict:
            gt_boxes_mask = data_dict["gt_boxes_mask"]
            data_dict["gt_boxes"] = data_dict["gt_boxes"][gt_boxes_mask]
            data_dict["gt_names"] = data_dict["gt_names"][gt_boxes_mask]
            data_dict.pop("gt_boxes_mask")
        return data_dict
