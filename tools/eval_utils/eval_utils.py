import datetime
import json
import os
import os.path as osp
import pickle
import time


# from src.bounding_box import BoundingBox
# from src.utils.enumerators import (BBFormat, BBType, CoordinatesType,
#                                    MethodAveragePrecision)
import numpy as np
import torch
import tqdm

# import src.evaluators.coco_evaluator as coco_evaluator
# import src.evaluators.pascal_voc_evaluator as pascal_voc_evaluator
# import src.utils.converter as converter
# import src.utils.general_utils as general_utils

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


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

    if ret_dict.get("uncer_type", None) is not None:
        uncer_picp = ret_dict.get("uncer_picp", 0)
        uncer_soften_mpiw = ret_dict.get("uncer_soft_mpiw", 0)
        uncer_mpiw = ret_dict.get("uncer_mpiw", 0)
        uncer_valid_total_cnt = ret_dict.get("valid_cnt", 1)

        if uncer_picp is not None and uncer_soften_mpiw is not None:
            uncer_picp = uncer_picp.detach().cpu().numpy()
            uncer_mpiw = uncer_mpiw.detach().cpu().numpy()
            uncer_soften_mpiw = uncer_soften_mpiw.detach().cpu().numpy()
            # print(uncer_picp, uncer_soften_mpiw)
            metric["uncer_valid_cnt"] = (
                metric.get("uncer_valid_cnt", 0) + uncer_valid_total_cnt
            )
            # uncer_cnt = uncer_picp.shape[-1]
            # for i in range(uncer_cnt):
            metric["uncer_picp"] = (
                metric.get("uncer_picp", 0) + uncer_picp * uncer_valid_total_cnt
            )
            metric["uncer_soften_mpiw"] = (
                metric.get("uncer_soften_mpiw", 0)
                + uncer_soften_mpiw * uncer_valid_total_cnt
            )
            metric["uncer_mpiw"] = (
                metric.get("uncer_mpiw", 0) + uncer_mpiw * uncer_valid_total_cnt
            )


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

    if metric.get("uncer_valid_cnt", None) is not None:
        # result contain uncer metrics
        picp = metric["uncer_picp"]
        mpiw = metric["uncer_mpiw"]
        soft_mpiw = metric["uncer_soften_mpiw"]
        for i in range(picp.shape[-1]):
            ret_dict[f"uncer/picp_{i + 1}"] = picp[..., i] / max(
                metric["uncer_valid_cnt"], 1
            )
            ret_dict[f"uncer/mpiw_{i + 1}"] = mpiw[..., i] / max(
                metric["uncer_valid_cnt"], 1
            )
            ret_dict[f"uncer/soften_mpiw_{i + 1}"] = soft_mpiw[..., i] / max(
                metric["uncer_valid_cnt"], 1
            )
        ret_dict["uncer/valid_num"] = metric["uncer_valid_cnt"]

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

    # keys = sorted(ret_dict.keys())
    # write_line = [ret_dict[key] for key in keys]
    # # print(keys)
    # if not (os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0):
    #     with open(csv_path, "w") as f:
    #         f.write(",".join(keys) + "\n")

    # NOTE 0 1 指 overlap 是哪一个
    # axis0: easy moderate hard
    # axis1: class name
    # overlap_0_7 = np.array(
    #     [
    #         [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
    #         [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
    #         [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
    #     ]
    # )
    # overlap_0_5 = np.array(
    #     [
    #         [0.7, 0.5, 0.5, 0.7, 0.5, 0.5],
    #         [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
    #         [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
    #     ]
    # )
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
        *[
            ret_dict[f"uncer/picp_{s}"]
            for s in range(1, 9)
            if f"uncer/picp_{s}" in ret_dict.keys()
        ],
        *[
            ret_dict[f"uncer/soften_mpiw_{s}"]
            for s in range(1, 9)
            if f"uncer/soften_mpiw_{s}" in ret_dict.keys()
        ],
        *[
            ret_dict[f"uncer/mpiw_{s}"]
            for s in range(1, 9)
            if f"uncer/mpiw_{s}" in ret_dict.keys()
        ],
    ]
    # if evaluate2d:
    #     write_line.extend(
    #         [
    #             ret_dict["report2d"]["AP50"],
    #             ret_dict["report_project"]["AP50"],
    #             ret_dict["only_in_3d"],
    #             ret_dict["only_in_2d"],
    #             ret_dict["both"],
    #             ret_dict["metric2d"]["1"]["FP"],
    #             ret_dict["metric2d"]["1"]["TP"],
    #             ret_dict["metric_project"]["1"]["FP"],
    #             ret_dict["metric_project"]["1"]["TP"],
    #             "",
    #             now,
    #             "",
    #             " ".join(sys.argv[1:]),
    #         ]
    #     )
    # else:
    #     write_line.extend(["", now, "", " ".join(sys.argv[1:])])

    # keys = (
    #     [
    #         "BEV-R40 Easy",
    #         "BEV-R40 Moderate",
    #         "BEV-R40 Hard",
    #         "3D-R40 Easy",
    #         "3D-R40 Moderate",
    #         "3D-R40 Hard",
    #         "BEV-R40 Easy05",
    #         "BEV-R40 Moderate05",
    #         "BEV-R40 Hard05",
    #         "3D-R40 Easy05",
    #         "3D-R40 Moderate05",
    #         "3D-R40 Hard05",
    #         "2D-mAP@0.5-Image",
    #         "2D-mAP@0.5-Project",
    #     ]
    #     if evaluate2d
    #     else [
    #         "BEV-R40 Easy",
    #         "BEV-R40 Moderate",
    #         "BEV-R40 Hard",
    #         "3D-R40 Easy",
    #         "3D-R40 Moderate",
    #         "3D-R40 Hard",
    #         "BEV-R40 Easy05",
    #         "BEV-R40 Moderate05",
    #         "BEV-R40 Hard05",
    #         "3D-R40 Easy05",
    #         "3D-R40 Moderate05",
    #         "3D-R40 Hard05",
    #     ]
    # )

    # tb_dict = {keys[i]: write_line[i] for i in range(len(keys))}
    # tb_dict["BEV-R40 Avg"] = sum(write_line[:3]) / 3
    # tb_dict["3D-R40 Avg"] = sum(write_line[3:6]) / 3
    # tb_dict["Avg"] = sum(write_line[:6]) / 6
    write_line = [(("%.4f" % x) if isinstance(x, float) else x) for x in write_line]
    # write_line = [(("%d" % x) if isinstance(x, int) else x) for x in write_line]

    tb_dict = ret_dict
    with open(csv_path, "a") as f:
        f.write(",".join(write_line) + "\n")

    logger.info("Result is save to %s" % result_dir)
    logger.info("****************Evaluation done.*****************")
    return tb_dict, write_line


if __name__ == "__main__":
    pass
