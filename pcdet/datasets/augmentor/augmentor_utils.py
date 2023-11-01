import copy

import numpy as np

from pcdet.utils import debug_utils

from ...ops.iou3d_nms import iou3d_nms_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils


def random_flip_along_x(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]

    return gt_boxes, points


def random_flip_points_and_image(gt_boxes, points, gt_boxes2d, image, calib):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
        image: (H, W, 3)
        calib: Calibration
    Returns:
    """
    # debug_utils.save_image_boxes_and_pts(image, gt_boxes2d, points[:,:3], calib,
    #                                      img_name='org.jpg')

    pts_enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if pts_enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]

        pts_reverse = np.identity(4, dtype=calib.V2C.dtype)
        pts_reverse[1, 1] = -1
        calib.V2C = calib.V2C @ pts_reverse

    img_enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if img_enable:
        h, w, _ = image.shape
        image = np.ascontiguousarray(image[:, ::-1, :])

        x_min, y_min, x_max, y_max = np.split(gt_boxes2d, 4, axis=1)
        gt_boxes2d = np.concatenate(
            [w - x_max, y_min, w - x_min, y_max], axis=1
        ).astype(gt_boxes2d.dtype)

        P = np.asarray([[-1.0, 0.0, w], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        calib.P2 = P @ calib.P2

    # debug_utils.save_image_boxes_and_pts(image, gt_boxes2d, points[:,:3], calib,
    #                                      img_name='aug.jpg')

    return gt_boxes, points, gt_boxes2d, image, calib


def random_flip_along_y(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]

    return gt_boxes, points


def global_rotation(gt_boxes, points, rot_range):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = common_utils.rotate_points_along_z(
        points[np.newaxis, :, :], np.array([noise_rotation])
    )[0]
    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(
        gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation])
    )[0]
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[
                np.newaxis, :, :
            ],
            np.array([noise_rotation]),
        )[0][:, 0:2]

    return gt_boxes, points


def global_fusion_rotation(gt_boxes, points, calib, rot_range):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max],
        calib: Calibration
    Returns:
    """
    # debug_utils.save_image_boxes_and_pts(image, None, points[:,:3], calib,
    #                                      img_name='org.jpg')
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])

    cosa = np.cos(-noise_rotation)
    sina = np.sin(-noise_rotation)
    inverse_rot_matrix = np.asarray(
        [
            [cosa, -sina, 0.0, 0.0],
            [sina, cosa, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=calib.V2C.dtype,
    )
    calib.V2C = calib.V2C @ inverse_rot_matrix

    points = common_utils.rotate_points_along_z(
        points[np.newaxis, :, :], np.array([noise_rotation])
    )[0]
    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(
        gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation])
    )[0]
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[
                np.newaxis, :, :
            ],
            np.array([noise_rotation]),
        )[0][:, 0:2]

    # debug_utils.save_image_boxes_and_pts(image, None, points[:, :3], calib,
    #                                      img_name='aug.jpg')

    return gt_boxes, points, calib


def global_scaling(gt_boxes, points, scale_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    return gt_boxes, points


def global_fusion_scaling(gt_boxes, points, calib, scale_range):
    """
    Args:
        calib: Calibration
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    # debug_utils.save_image_boxes_and_pts(image, None, points[:,:3], calib,
    #                                      img_name='org.jpg')
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    inverse_scale = 1.0 / noise_scale
    inverse_scale_matrix = np.asarray(
        [
            [inverse_scale, 0.0, 0.0, 0.0],
            [0, inverse_scale, 0.0, 0.0],
            [0.0, 0.0, inverse_scale, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=calib.V2C.dtype,
    )
    calib.V2C = calib.V2C @ inverse_scale_matrix

    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale

    # debug_utils.save_image_boxes_and_pts(image, None, points[:, :3], calib,
    #                                      img_name='aug.jpg')
    return gt_boxes, points, calib



def scale_pre_object(gt_boxes, points, gt_boxes_mask, scale_perturb, num_try=50):
    """
    uniform sacle object with given range
    Args:
        gt_boxes: (N, 7) under unified coordinates
        points: (M, 3 + C) points in lidar
        gt_boxes_mask: (N), boolen mask for
        scale_perturb:
        num_try:
    Returns:
    """
    num_boxes = gt_boxes.shape[0]
    if not isinstance(scale_perturb, (list, tuple, np.ndarray)):
        scale_perturb = [-scale_perturb, scale_perturb]

    # boxes wise scale ratio
    scale_noises = np.random.uniform(scale_perturb[0], scale_perturb[1], size=[num_boxes, num_try])
    for k in range(num_boxes):
        if gt_boxes_mask[k] == 0:
            continue

        scl_box = copy.deepcopy(gt_boxes[k])
        scl_box = scl_box.reshape(1, -1).repeat([num_try], axis=0)
        scl_box[:, 3:6] = scl_box[:, 3:6] * scale_noises[k].reshape(-1, 1).repeat([3], axis=1)

        # detect conflict
        # [num_try, N-1]
        if num_boxes > 1:
            self_mask = np.ones(num_boxes, dtype=np.bool_)
            self_mask[k] = False
            iou_matrix = iou3d_nms_utils.boxes_bev_iou_cpu(scl_box, gt_boxes[self_mask])
            ious = np.max(iou_matrix, axis=1)
            no_conflict_mask = (ious == 0)
            # all trys have conflict with other gts
            if no_conflict_mask.sum() == 0:
                continue

            # scale points and assign new box
            try_idx = no_conflict_mask.nonzero()[0][0]
        else:
            try_idx = 0

        point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(
            points[:, 0:3],np.expand_dims(gt_boxes[k], axis=0)).squeeze(0)

        obj_points = points[point_masks > 0]
        obj_center, lwh, ry = gt_boxes[k, 0:3], gt_boxes[k, 3:6], gt_boxes[k, 6]

        # relative coordinates
        obj_points[:, 0:3] -= obj_center
        obj_points = common_utils.rotate_points_along_z(np.expand_dims(obj_points, axis=0), -ry).squeeze(0)
        new_lwh = lwh * scale_noises[k][try_idx]

        obj_points[:, 0:3] = obj_points[:, 0:3] * scale_noises[k][try_idx]
        obj_points = common_utils.rotate_points_along_z(np.expand_dims(obj_points, axis=0), ry).squeeze(0)
        # calculate new object center to avoid object float over the road
        obj_center[2] += (new_lwh[2] - lwh[2]) / 2
        obj_points[:, 0:3] += obj_center
        points[point_masks > 0] = obj_points
        gt_boxes[k, 3:6] = new_lwh

        # if enlarge boxes, remove bg points
        if scale_noises[k][try_idx] > 1:
            points_dst_mask = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 0:3],
                                                                        np.expand_dims(gt_boxes[k],
                                                                                       axis=0)).squeeze(0)

            keep_mask = ~np.logical_xor(point_masks, points_dst_mask)
            points = points[keep_mask]

    return points, gt_boxes
