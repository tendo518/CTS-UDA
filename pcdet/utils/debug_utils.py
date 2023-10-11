import numpy as np
from skimage import io
import uuid
import cv2
import os.path as osp
import os
import pcdet.utils.box_utils as box_utils
from pcdet.structures import ImageList
import torch
import matplotlib.pylab as plt
from .box_utils import boxes_to_corners_3d

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pickle


def save_image(image, name="test.jpg"):
    io.imsave(f"../debug/{name}", image)


edges = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [0, 4],
    [1, 5],
    [5, 4],
    [2, 6],
    [5, 6],
    [7, 6],
    [7, 3],
    [7, 4],
]


def save_image_with_boxes(
    img, bboxes=None, img_name=None, debug="../debug", save=True, color=None
):
    color = color or (0, 255, 0)
    if img_name is None:
        img_name = str(uuid.uuid1()) + ".png"

    if bboxes is not None:
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            if len(bbox.shape) == 1:
                # 2d box
                x_min = bbox[0]
                y_min = bbox[1]
                x_max = bbox[2]
                y_max = bbox[3]
                cv2.rectangle(
                    img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 3
                )
            else:
                # 3d box
                for start, end in edges:
                    p1 = bbox[start]
                    p2 = bbox[end]
                    cv2.line(
                        img,
                        (int(p1[0]), int(p1[1])),
                        (int(p2[0]), int(p2[1])),
                        color,
                        1,
                    )

    img = img.astype(np.uint8)
    if save:
        io.imsave(osp.join(debug, img_name), img)
    return img


def save_image_boxes_and_pts(
    img, bboxes, pts, calib, *, img_name=None, debug="../debug"
):
    img = np.ascontiguousarray(img)
    img_pts = calib.lidar_to_img(pts)[0].astype(np.int)
    img = img.copy()
    img_pts[:, 0] = np.clip(img_pts[:, 0], 0, img.shape[1] - 1)
    img_pts[:, 1] = np.clip(img_pts[:, 1], 0, img.shape[0] - 1)
    img[img_pts[:, 1], img_pts[:, 0], :] = (255, 0, 0)
    img = img.astype(np.uint8)
    save_image_with_boxes(img, bboxes, img_name, debug)


def save_image_boxes_and_pts_labels_and_mask(
    img,
    bboxes,
    pts,
    pts_label,
    calib,
    pred_masks2d=[],
    *,
    img_name=None,
    debug="../debug",
):
    img = np.ascontiguousarray(img)
    img_pts = calib.lidar_to_img(pts)[0].astype(np.int)
    img = img.copy()
    img_pts[:, 0] = np.clip(img_pts[:, 0], 0, img.shape[1] - 1)
    img_pts[:, 1] = np.clip(img_pts[:, 1], 0, img.shape[0] - 1)
    img[img_pts[:, 1], img_pts[:, 0], :] = (255, 0, 0)

    img_pts[:, 0] = np.clip(img_pts[:, 0], 0, img.shape[1] - 1)
    img_pts[:, 1] = np.clip(img_pts[:, 1], 0, img.shape[0] - 1)
    img[img_pts[:, 1], img_pts[:, 0], :] = (255, 0, 0)
    max_cls = pts_label.max() + 1
    colors = [
        (0, 255, 127),
        (0, 191, 255),
        (255, 255, 0),
        (255, 127, 80),
        (205, 92, 92),
    ]
    for j in range(1, max_cls):
        cls_ids = pts_label == j
        img[img_pts[cls_ids, 1], img_pts[cls_ids, 0], :] = colors[j % len(colors)]

    if len(pred_masks2d) > 0:
        pred_masks2d = np.max(pred_masks2d, axis=0) > 0
        img[pred_masks2d] = img[pred_masks2d] * 0.5 + 122
        img = np.clip(img, 0, 255)

    img = img.astype(np.uint8)
    save_image_with_boxes(img, bboxes, img_name, debug)


def save_image_with_instances(
    data_batch=None, std=[1.0, 1.0, 1.0], mean=[103.53, 116.28, 123.675]
):
    images = data_batch["images"]
    if isinstance(images, ImageList):
        images = images.tensor
    instances = data_batch["instances"]
    frame_ids = data_batch["frame_id"]
    proposals = data_batch.get("proposals", None)

    std = np.asarray(std)[None, None, :]
    mean = np.asarray(mean)[None, None, :]

    pts_img = data_batch["pts_img"]
    pts_batch_ids = data_batch["point_coords"][:, 0]
    pts_label = data_batch.get("pts_fake_target")
    colors = [
        (0, 255, 127),
        (0, 191, 255),
        (255, 255, 0),
        (255, 127, 80),
        (205, 92, 92),
    ]

    for i, (image, inst) in enumerate(zip(images, instances)):
        image = image.cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = np.ascontiguousarray(image)
        image = std * image + mean
        bbox = inst.get("gt_boxes").tensor.cpu().numpy()
        fid = (frame_ids[i] + ".png") if frame_ids is not None else None
        image = save_image_with_boxes(
            img=image, bboxes=bbox, img_name=fid, save=(data_batch is None)
        )
        if "pred_mask2d" in data_batch:
            pred_masks2d = data_batch["pred_mask2d"][i].cpu().numpy()
            if len(pred_masks2d) > 0:
                pred_masks2d = np.max(pred_masks2d, axis=0) > 0
                image[pred_masks2d] = image[pred_masks2d] * 0.5 + 122
                image = np.clip(image, 0, 255)

        if pts_label is not None:
            single_pts_label = pts_label[pts_batch_ids == i].cpu().numpy()
            single_pts_img = pts_img[pts_batch_ids == i].cpu().numpy().astype(np.int)
            single_pts_img = single_pts_img[:, 1:]
            single_pts_img[:, 0] = np.clip(single_pts_img[:, 0], 0, image.shape[1] - 1)
            single_pts_img[:, 1] = np.clip(single_pts_img[:, 1], 0, image.shape[0] - 1)
            image[single_pts_img[:, 1], single_pts_img[:, 0], :] = (255, 0, 0)
            max_cls = single_pts_label.max() + 1
            for j in range(1, max_cls):
                cls_ids = single_pts_label == j
                image[
                    single_pts_img[cls_ids, 1], single_pts_img[cls_ids, 0], :
                ] = colors[j]

            image = image.astype(np.uint8)

        if proposals is not None:
            bbox = proposals[i].get("proposal_boxes").tensor.cpu().numpy()
            logits = proposals[i].get("objectness_logits").cpu().numpy()
            top50_indices = np.argsort(logits)[:50]
            top50_bbox = bbox[top50_indices]
            image = save_image_with_boxes(
                img=image,
                bboxes=top50_bbox,
                img_name=fid,
                save=(data_batch is None),
                color=(0, 0, 255),
            )

        if data_batch is not None:
            calib = data_batch["calib"][i]
            boxes3d = data_batch["gt_boxes"][i] if "gt_boxes" in data_batch else []
            j = len(boxes3d) - 1
            while j >= 0 and boxes3d[j].sum() == 0:
                j -= 1
            if j >= 0:
                boxes3d = boxes3d[: j + 1]
                boxes2d, boxes2d_corners = box_utils.lidar_box_to_image_box(
                    boxes3d, calib
                )
                save_image_with_boxes(
                    img=image,
                    bboxes=boxes2d_corners,
                    img_name=fid,
                    save=True,
                    color=(255, 0, 0),
                )
            else:
                save_image_with_boxes(
                    img=image, bboxes=None, img_name=fid, save=True, color=(255, 0, 0)
                )


def make_poly3dcollection(box: np.ndarray, c):
    """

    Args:
        box: 8 x 3

    Returns:

    """
    box = box[:, :3]
    verts = [
        [box[0], box[1], box[2], box[3]],
        [box[4], box[5], box[6], box[7]],
        [box[0], box[1], box[5], box[4]],
        [box[2], box[3], box[7], box[6]],
        [box[1], box[2], box[6], box[5]],
        [box[4], box[7], box[3], box[0]],
    ]
    return Poly3DCollection(verts, linewidths=1, edgecolors=c, facecolors=c, alpha=0.10)


def show_roi(
    roi_features: torch.Tensor,
    rois: torch.Tensor,
    apply_aug_boxes: torch.Tensor,
    aug_roi_features: torch.Tensor,
    aug_rois: torch.Tensor,
    backout_aug_boxes: torch,
    roi_aug_dict,
    reg_valid_mask,
    org_rec_pts=None,
    aug_rec_pts=None,
    n=10,
    fig_size=5,
):
    """

    Args:
        roi_features: (B * N) x (3 + C)
        rois: B x N x 7

    Returns:

    """
    reg_valid_mask = reg_valid_mask > 0
    roi_features = roi_features[reg_valid_mask.view(-1)]
    rois = rois[reg_valid_mask]
    apply_aug_boxes = apply_aug_boxes[reg_valid_mask]

    aug_roi_features = aug_roi_features[reg_valid_mask.view(-1)]
    aug_rois = aug_rois[reg_valid_mask]
    backout_aug_boxes = backout_aug_boxes[reg_valid_mask]

    if org_rec_pts is not None:
        org_rec_pts = org_rec_pts[reg_valid_mask.view(-1)].cpu().detach().numpy()
        aug_rec_pts = aug_rec_pts[reg_valid_mask.view(-1)].cpu().detach().numpy()

    alpha = roi_aug_dict["alpha"][reg_valid_mask].cpu().numpy()
    scale = roi_aug_dict["scale"][reg_valid_mask].cpu().numpy()
    delta_xyz = roi_aug_dict["delta_xyz"][reg_valid_mask].cpu().numpy()
    flip = roi_aug_dict["flip"]

    roi_features = roi_features.cpu().detach().numpy()
    rois = boxes_to_corners_3d(rois.view(-1, rois.shape[-1])).cpu().numpy()
    apply_aug_boxes = (
        boxes_to_corners_3d(apply_aug_boxes.view(-1, apply_aug_boxes.shape[-1]))
        .cpu()
        .numpy()
    )

    aug_roi_features = aug_roi_features.cpu().detach().numpy()
    aug_rois = boxes_to_corners_3d(aug_rois.view(-1, aug_rois.shape[-1])).cpu().numpy()
    backout_aug_boxes = (
        boxes_to_corners_3d(backout_aug_boxes.view(-1, backout_aug_boxes.shape[-1]))
        .cpu()
        .numpy()
    )

    x_lim = (-3, 3)
    y_lim = (-3, 3)
    z_lim = (-3, 3)
    start_idx = len(os.listdir("../debug"))
    for i in range(n):
        c_org = "r"
        points_org = roi_features[i, :, :]
        points_org = points_org[points_org.sum(1) != 0]
        box_org = rois[i, :, :3]
        x1 = box_org[1] - box_org[0]

        fig = plt.figure(figsize=(3 * fig_size, 2 * fig_size))
        ax1 = fig.add_subplot(2, 3, 1, projection="3d")

        ax1.scatter(points_org[:, 0], points_org[:, 1], points_org[:, 2], c=c_org)
        ax1.add_collection3d(make_poly3dcollection(box_org, c_org))
        ax1.add_collection3d(make_poly3dcollection(backout_aug_boxes[i, :, :3], "cyan"))

        ax1.set_xlim(x_lim)
        ax1.set_ylim(y_lim)
        ax1.set_zlim(z_lim)
        ax1.view_init(elev=0, azim=0)
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")
        ax1.set_title("point num=%d" % len(points_org))

        ax2 = fig.add_subplot(2, 3, 2, projection="3d")
        ax2.scatter(
            points_org[:, 0], points_org[:, 1], points_org[:, 2], c=c_org, marker="x"
        )
        ax2.add_collection3d(make_poly3dcollection(box_org, c_org))

        c_aug = "b"
        points_aug = aug_roi_features[
            i,
            :,
        ]
        points_aug = points_aug[points_aug.sum(1) != 0]
        box_aug = aug_rois[i, :, :3]

        x2 = box_aug[1] - box_aug[0]
        ax2.scatter(points_aug[:, 0], points_aug[:, 1], points_aug[:, 2], c=c_aug)
        ax2.add_collection3d(make_poly3dcollection(box_aug, c_aug))
        ax2.set_xlim(x_lim)
        ax2.set_ylim(y_lim)
        ax2.set_zlim(z_lim)
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")
        ax2.view_init(elev=90, azim=0)
        #
        ax3 = fig.add_subplot(2, 3, 3, projection="3d")
        ax3.scatter(points_aug[:, 0], points_aug[:, 1], points_aug[:, 2], c=c_aug)
        ax3.add_collection3d(make_poly3dcollection(box_aug, c_aug))

        ax3.add_collection3d(make_poly3dcollection(apply_aug_boxes[i, :, :3], "cyan"))

        ax3.set_xlim(x_lim)
        ax3.set_ylim(y_lim)
        ax3.set_zlim(z_lim)
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_zlabel("z")
        ax3.view_init(elev=0, azim=90)
        ax3.set_title("point num=%d" % len(points_aug))

        theta = np.sum(x1 * x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        theta = np.arccos(theta)
        fig.suptitle(
            "alpha=%.4f, scale=(%.4f, %.4f, %.4f), "
            "offset=(%.4f, %.4f, %.4f), flip=%s, "
            "alpha_mod=%.4f, theta_mod=%.4f"
            % (
                alpha[i],
                *scale[i],
                *delta_xyz[i],
                str(flip.item()),
                alpha[i] % (2 * np.pi),
                theta % (2 * np.pi),
            )
        )

        if org_rec_pts is not None:
            ax = fig.add_subplot(2, 3, 4, projection="3d")
            points_rec = org_rec_pts[i, :, :]
            ax.scatter(
                points_org[:, 0], points_org[:, 1], points_org[:, 2], label="Org"
            )
            ax.scatter(
                points_rec[:, 0], points_rec[:, 1], points_rec[:, 2], label="Rec"
            )
            plt.legend()
            ax.view_init(elev=90, azim=0)
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_zlim(z_lim)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax = fig.add_subplot(2, 3, 5, projection="3d")

            points_rec_aug = aug_rec_pts[i, :, :]
            ax.scatter(
                points_aug[:, 0], points_aug[:, 1], points_aug[:, 2], label="Aug Org"
            )
            ax.scatter(
                points_rec_aug[:, 0],
                points_rec_aug[:, 1],
                points_rec_aug[:, 2],
                label="Aug Rec",
            )
            ax.view_init(elev=90, azim=0)
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_zlim(z_lim)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            plt.legend()

        with open("../debug/pkl/fig%d.pkl" % (i + start_idx), "wb") as f:
            pickle.dump(fig, f)
        plt.savefig(f"../debug/{i + start_idx}.png")
