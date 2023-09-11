import pickle
import time
import datetime
import os
import os.path as osp
import sys

import numpy as np
import torch
import tqdm
import json

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


# import src.evaluators.coco_evaluator as coco_evaluator
# import src.evaluators.pascal_voc_evaluator as pascal_voc_evaluator
# import src.utils.converter as converter
# import src.utils.general_utils as general_utils

# from src.bounding_box import BoundingBox
# from src.utils.enumerators import (BBFormat, BBType, CoordinatesType,
#                                    MethodAveragePrecision)


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric["recall_roi_%s" % str(cur_thresh)] += ret_dict.get(
            "roi_%s" % str(cur_thresh), 0
        )
        metric["recall_rcnn_%s" % str(cur_thresh)] += ret_dict.get(
            "rcnn_%s" % str(cur_thresh), 0
        )
    metric["gt_num"] += ret_dict.get("gt", 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict["recall_%s" % str(min_thresh)] = "(%d, %d) / %d" % (
        metric["recall_roi_%s" % str(min_thresh)],
        metric["recall_rcnn_%s" % str(min_thresh)],
        metric["gt_num"],
    )

    if ret_dict.get("picp", None) is not None:
        disp_dict


def convert_dict(data: dict) -> dict:
    new_dict = {}
    for key, value in data.items():
        if not isinstance(key, (np.int32, bool, str, float)):
            key = str(key)
        if isinstance(value, dict):
            value = convert_dict(value)
        elif isinstance(value, np.ndarray):
            if value.dtype == np.int64:
                value = value.astype(np.int32)
            value = value.tolist()
        elif isinstance(value, np.int64):
            value = int(value)
        new_dict[key] = value
    return new_dict


def eval_one_epoch(
    cfg,
    model,
    dataloader,
    epoch_id,
    logger,
    dist_test=False,
    save_to_file=False,
    result_dir=None,
    save_preds=False,
):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / "final_result" / "data"

    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        "gt_num": 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric["recall_roi_%s" % str(cur_thresh)] = 0
        metric["recall_rcnn_%s" % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info("*************** EPOCH %s EVALUATION *****************" % epoch_id)

    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False
        )

    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(
            total=len(dataloader), leave=True, desc="eval", dynamic_ncols=True
        )
    start_time = time.time()
    evaluate2d = False
    debug = cfg.get("DEBUG", False)
    save_dict = {}
    if debug:
        debug_output_dir = result_dir / "debug"
        debug_output_dir.mkdir(parents=True, exist_ok=True)

    if os.path.exists(result_dir / "result.pkl"):
        with open(result_dir / "result.pkl", "rb") as f:
            det_annos = pickle.load(f)
    else:
        for i, batch_dict in enumerate(dataloader):
            if debug and i > 50:
                break
            load_data_to_gpu(batch_dict)
            with torch.no_grad():
                pred_dicts, ret_dict = model(batch_dict)
            disp_dict = {}
            # if pred_dicts[0].get("pred_aus") is not None:
            #     # TODO EVAL uncertainity with PICP
            #     for batch_index in range(batch_dict["batch_size"]):
            #         pred_aus = pred_dicts[batch_index]["pred_aus"]
            #         pred_boxes  = pred_dicts[batch_index]["pred_boxes"]

            #         au_code_size = pred_aus.shape[-1]
            #         if au_code_size == 8:
            #             # au for corners
            #             pass
            #         elif au_code_size == 7:
            #             # au for xyzwhla
            #             pass
            #         else:
            #             raise NotImplementedError

            # if pred_dicts[0].get('pred_boxes2d') is not None:
            #     batch_gt_boxes2d = [x.get('gt_boxes').tensor.cpu().numpy() for x in
            #                         batch_dict['instances']]
            #     batch_gt_labels2d = [x.get('gt_classes').cpu().numpy() for x in
            #                          batch_dict['instances']]
            #     evaluate2d = True
            #     for batch_index in range(batch_dict['batch_size']):

            #         image_shape = batch_dict['image_shape'][batch_index]
            #         new_shape = batch_dict['images'].image_sizes[batch_index]

            #         pred_boxes2d = pred_dicts[batch_index]['pred_boxes2d']
            #         pred_scores2d = pred_dicts[batch_index]['pred_scores2d']
            #         pred_labels2d = pred_dicts[batch_index]['pred_labels2d']
            #         pred_masks2d = pred_dicts[batch_index]['pred_masks2d']

            #         pred_boxes3d = pred_dicts[batch_index]['pred_boxes']
            #         pred_scores3d = pred_dicts[batch_index]['pred_scores']
            #         pred_labels3d = pred_dicts[batch_index]['pred_labels']

            #         if 'only_in_3d' in pred_dicts[batch_index]:
            #             only_in_3d += pred_dicts[batch_index]['only_in_3d']
            #             only_in_2d += pred_dicts[batch_index]['only_in_2d']
            #             both += pred_dicts[batch_index]['both']

            #         if 'only2d_ious' in pred_dicts[batch_index]:
            #             only2d_ious.extend(pred_dicts[batch_index]['only2d_ious'])
            #             only3d_ious.extend(pred_dicts[batch_index]['only3d_ious'])
            #             both_ious.extend(pred_dicts[batch_index]['both_ious'])

            #         gt_boxes2d = batch_gt_boxes2d[batch_index]
            #         gt_labels2d = batch_gt_labels2d[batch_index]
            #         frame_id = batch_dict['frame_id'][batch_index]
            #         calib = batch_dict['calib'][batch_index]

            #         # if 'both' not in pred_dicts[batch_index]:
            #         save_dict[frame_id] = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k,v in pred_dicts[batch_index].items() if k != 'pred_masks2d'}

            #         img = dataset.get_image(frame_id) if debug else None

            #         for det_box, det_label, det_score in zip(pred_boxes2d, pred_labels2d, pred_scores2d):
            #             det_box = tuple(det_box)
            #             det_bbs.append(
            #                 BoundingBox(
            #                     image_name=frame_id,
            #                     class_id=str(det_label),
            #                     coordinates=det_box,
            #                     type_coordinates=CoordinatesType.ABSOLUTE,
            #                     img_size=None,
            #                     bb_type=BBType.DETECTED,
            #                     confidence=det_score,
            #                     format=BBFormat.XYX2Y2
            #                 )
            #             )
            #             if img is not None and det_bbs[-1].get_confidence() > 0.5:
            #                 img = general_utils.add_bb_into_image(img,
            #                                                       det_bbs[-1],
            #                                                       color=(255, 0, 0),
            #                                                       label=f'det:{det_label}')
            #         pred_project,_ = box_utils.lidar_box_to_image_box(boxes3d=pred_boxes3d,
            #                                                        calib=calib)
            #         pred_project = box_utils.recover_boxes_2d(pred_project,
            #                                                      image_shape,
            #                                                      new_shape)
            #         for det_box, det_label, det_score in zip(pred_project,
            #                                                  pred_labels3d.cpu().numpy(),
            #                                                  pred_scores3d.cpu().numpy()):
            #             det_box = tuple(det_box.cpu().numpy())
            #             det_project_bbs.append(
            #                 BoundingBox(
            #                     image_name=frame_id,
            #                     class_id=str(det_label),
            #                     coordinates=det_box,
            #                     type_coordinates=CoordinatesType.ABSOLUTE,
            #                     img_size=None,
            #                     bb_type=BBType.DETECTED,
            #                     confidence=det_score,
            #                     format=BBFormat.XYX2Y2
            #                 )
            #             )
            #             if img is not None and det_project_bbs[
            #                 -1].get_confidence() > 0.5:
            #                 img = general_utils.add_bb_into_image(img,
            #                                                       det_project_bbs[-1],
            #                                                       color=(
            #                                                       0, 255, 0),
            #                                                       label=f'3d:{det_label}')

            #         for gt_box, gt_label in zip(gt_boxes2d, gt_labels2d):
            #             gt_box = tuple(gt_box)
            #             gt_bbs.append(
            #                 BoundingBox(
            #                     image_name=frame_id,
            #                     class_id=str(gt_label),
            #                     coordinates=gt_box,
            #                     type_coordinates=CoordinatesType.ABSOLUTE,
            #                     img_size=None,
            #                     bb_type=BBType.GROUND_TRUTH,
            #                     confidence=None,
            #                     format=BBFormat.XYX2Y2
            #                 )
            #             )
            #             if img is not None:
            #                 img = general_utils.add_bb_into_image(img,
            #                                                       gt_bbs[-1],
            #                                                       color=(0, 0, 255),
            #                                                       label=f'gt:{gt_label}')
            #         if img is not None:
            #             output_path = debug_output_dir / (frame_id + '.png')
            #             # io.imsave(output_path, dataset.get_image(frame_id))
            #             if len(pred_masks2d) > 0:
            #                 pred_masks2d = np.max(pred_masks2d, axis=0) > 0
            #                 img[pred_masks2d] = img[pred_masks2d] * 0.5 + 122
            #                 img = np.clip(img, 0, 255)
            #             io.imsave(output_path, img[:,:,::-1])

            statistics_info(cfg, ret_dict, metric, disp_dict)
            annos = dataset.generate_prediction_dicts(
                batch_dict,
                pred_dicts,
                class_names,
                output_path=final_output_dir if save_to_file else None,
            )
            det_annos += annos

            if cfg.LOCAL_RANK == 0:
                progress_bar.set_postfix(disp_dict)
                progress_bar.update()

    # plt.bar(['only3d', 'only2d', 'both'], [only_in_3d, only_in_2d, both])
    # plt.ylabel('Count')
    # plt.show()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(
            det_annos, len(dataset), tmpdir=result_dir / "tmpdir"
        )
        metric = common_utils.merge_results_dist(
            [metric], world_size, tmpdir=result_dir / "tmpdir"
        )

    logger.info("*************** Performance of EPOCH %s *****************" % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info(
        "Generate label finished(sec_per_example: %.4f second)." % sec_per_example
    )

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric["gt_num"]
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric["recall_roi_%s" % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric["recall_rcnn_%s" % str(cur_thresh)] / max(
            gt_num_cnt, 1
        )
        logger.info("recall_roi_%s: %f" % (cur_thresh, cur_roi_recall))
        logger.info("recall_rcnn_%s: %f" % (cur_thresh, cur_rcnn_recall))
        ret_dict["recall/roi_%s" % str(cur_thresh)] = cur_roi_recall
        ret_dict["recall/rcnn_%s" % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno["name"].__len__()
    logger.info(
        "Average predicted number of objects(%d samples): %.3f"
        % (len(det_annos), total_pred_objects / max(1, len(det_annos)))
    )

    with open(result_dir / "result.pkl", "wb") as f:
        pickle.dump(det_annos, f)
    if save_preds:
        with open(result_dir / "predict.pkl", "wb") as f:
            pickle.dump(save_dict, f)

    result_str, result_dict = dataset.evaluation(
        det_annos,
        class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir,
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    # # TODO save 2d results
    # if evaluate2d:
    #     report2d = coco_evaluator.get_coco_summary(gt_bbs, det_bbs)
    #     metrics_2d = coco_evaluator.get_coco_metrics(gt_bbs, det_bbs)

    #     logger.info('****************Evaluation 2D.*******************')
    #     formmat_string = '{:<15}{:<.4f}'
    #     for k,v in report2d.items():
    #         logger.info(formmat_string.format(k,v))
    #     print()
    #     pprint.pprint(metrics_2d)

    #     report_project = coco_evaluator.get_coco_summary(gt_bbs, det_project_bbs)
    #     metrics_project = coco_evaluator.get_coco_metrics(gt_bbs, det_project_bbs)

    #     logger.info('****************Evaluation Projection Boxes.*******************')
    #     formmat_string = '{:<15}{:<.4f}'
    #     for k, v in report_project.items():
    #         logger.info(formmat_string.format(k, v))
    #     print()
    #     pprint.pprint(metrics_project)
    #     ret_dict.update({
    #         'report2d':report2d,
    #         'metric2d':convert_dict(metrics_2d),
    #         'report_project':report_project,
    #         'metric_project':convert_dict(metrics_project),
    #         'only_in_3d': only_in_3d,
    #         'only_in_2d': only_in_2d,
    #         'both': both,
    #         'only3d_ious': only3d_ious,
    #         'only2d_ious': only2d_ious,
    #         'both_ious': both_ious
    #     })

    # iou_type = ['only_in_3d'] * len(only3d_ious)
    # iou_type.extend(['only_in_2d'] * len(only2d_ious))
    # iou_type.extend(['both'] * len(both_ious))
    # dataframe = {
    #     'iou_type': iou_type,
    #     'iou': np.concatenate([only3d_ious, only2d_ious, both_ious],
    #                           axis=0),
    # }

    # sns.histplot(dataframe, x="iou", hue="iou_type", element="step")
    #
    # plt.show()
    # print('only3d/only2d/both:%d,%d,%d' % (only_in_3d, only_in_2d, both))

    with open(result_dir / "result_json.json", "w") as f:
        json.dump(ret_dict, f)

    now = datetime.datetime.now() + datetime.timedelta(hours=8)
    now = now.strftime("%Y-%m-%d-%H-%M-%S")
    report_path = "../output/kitti_models/reports"
    os.makedirs(report_path, exist_ok=True)
    json_path = osp.join(report_path, now)
    os.makedirs(json_path, exist_ok=True)
    with open(osp.join(json_path, "result_json.json"), "w") as f:
        json.dump(ret_dict, f)
    csv_path = osp.join(report_path, "report.csv")
    write_line = [
        ret_dict["Car_bev/easy_R40_0.7_0"],
        ret_dict["Car_bev/moderate_R40_0.7_0"],
        ret_dict["Car_bev/hard_R40_0.7_0"],
        ret_dict["Car_3d/easy_R40_0.7_0"],
        ret_dict["Car_3d/moderate_R40_0.7_0"],
        ret_dict["Car_3d/hard_R40_0.7_0"],
        ret_dict["Car_bev/easy_R40_0.7_1"],
        ret_dict["Car_bev/moderate_R40_0.5_1"],
        ret_dict["Car_bev/hard_R40_0.5_1"],
        ret_dict["Car_3d/easy_R40_0.7_1"],
        ret_dict["Car_3d/moderate_R40_0.5_1"],
        ret_dict["Car_3d/hard_R40_0.5_1"],
    ]
    if evaluate2d:
        write_line.extend(
            [
                ret_dict["report2d"]["AP50"],
                ret_dict["report_project"]["AP50"],
                ret_dict["only_in_3d"],
                ret_dict["only_in_2d"],
                ret_dict["both"],
                ret_dict["metric2d"]["1"]["FP"],
                ret_dict["metric2d"]["1"]["TP"],
                ret_dict["metric_project"]["1"]["FP"],
                ret_dict["metric_project"]["1"]["TP"],
                "",
                now,
                "",
                " ".join(sys.argv[1:]),
            ]
        )
    else:
        write_line.extend(["", now, "", " ".join(sys.argv[1:])])

    keys = (
        [
            "BEV-R40 Easy",
            "BEV-R40 Moderate",
            "BEV-R40 Hard",
            "3D-R40 Easy",
            "3D-R40 Moderate",
            "3D-R40 Hard",
            "BEV-R40 Easy05",
            "BEV-R40 Moderate05",
            "BEV-R40 Hard05",
            "3D-R40 Easy05",
            "3D-R40 Moderate05",
            "3D-R40 Hard05",
            "2D-mAP@0.5-Image",
            "2D-mAP@0.5-Project",
        ]
        if evaluate2d
        else [
            "BEV-R40 Easy",
            "BEV-R40 Moderate",
            "BEV-R40 Hard",
            "3D-R40 Easy",
            "3D-R40 Moderate",
            "3D-R40 Hard",
            "BEV-R40 Easy05",
            "BEV-R40 Moderate05",
            "BEV-R40 Hard05",
            "3D-R40 Easy05",
            "3D-R40 Moderate05",
            "3D-R40 Hard05",
        ]
    )

    tb_dict = {keys[i]: write_line[i] for i in range(len(keys))}
    tb_dict["BEV-R40 Avg"] = sum(write_line[:3]) / 3
    tb_dict["3D-R40 Avg"] = sum(write_line[3:6]) / 3
    tb_dict["Avg"] = sum(write_line[:6]) / 6
    write_line = [(("%.4f" % x) if isinstance(x, float) else x) for x in write_line]
    write_line = [(("%d" % x) if isinstance(x, int) else x) for x in write_line]

    with open(csv_path, "a") as f:
        f.write(",".join(write_line) + "\n")

    logger.info("Result is save to %s" % result_dir)
    logger.info("****************Evaluation done.*****************")
    return tb_dict, write_line


if __name__ == "__main__":
    pass
