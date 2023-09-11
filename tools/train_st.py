import argparse
import copy
import datetime
import glob
import math
import os
import time
from pathlib import Path
from test import repeat_eval_ckpt
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data as data
import tqdm
from itertools import tee
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import checkpoint_state, save_checkpoint

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu, model_fn_decorator
from pcdet.utils import common_utils
from pcdet.datasets.dataset import DatasetTemplate


def seed_dataset(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def uncertainty_mining(
    model: nn.Module,
    org_dataset: DatasetTemplate,
    org_loader: data.DataLoader,
    batch_size: int = 15,
    mining_portation: float = 0.3,
    mining_label_thresh: float = 0.5,
    random_choices_flag: bool = False,
    seed: int = 42,
) -> data.Dataset:
    full_dataset = copy.deepcopy(org_dataset)

    # ugly hacking ...
    full_dataset.training = False
    full_dataset.data_augmentor = None
    full_dataset.data_processor.training = False  # data_processor.py L16
    full_dataset.data_processor.mode = "test"  # data_processor.py L17

    model.eval()

    if random_choices_flag:
        # choose randomly 
        org_len = len(full_dataset)
        random_index = np.random.choice(org_len, size=int(mining_portation * org_len))
        subset = data.Subset(org_dataset, indices=random_index)
        print(f"subset length: {len(subset)}/{len(org_dataset)}")

        return subset

    sampler = data.SequentialSampler(full_dataset)
    stat_loader = data.DataLoader(
        full_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=org_loader.num_workers,
        shuffle=False,
        collate_fn=org_loader.collate_fn,
        drop_last=False,
        sampler=sampler,
    )

    print("start mining")
    results_uncertainty = []
    for current_iter, batch_dict in enumerate(stat_loader):
        load_data_to_gpu(batch_dict)
        batch_dict["domain_label"] = 1

        pred_dicts: list[Any]
        pred_dicts, _ = model(batch_dict)
        for in_batch_index in range(len(pred_dicts)):
            pred_uncer = pred_dicts[in_batch_index]["pred_uncer"]
            pred_scores = pred_dicts[in_batch_index]["pred_scores"]

            real_dataset_index = current_iter * batch_size + in_batch_index
            all_instances_uncer = torch.mean(pred_uncer, dim=-1)
            all_instances_uncer = all_instances_uncer[pred_scores > mining_label_thresh]

            sample_uncer = torch.mean(all_instances_uncer, dim=-1).cpu().item()
            results_uncertainty.append(sample_uncer)
            print(
                f"\rframe_id: {batch_dict['frame_id'][in_batch_index]}\tindex: {real_dataset_index:05d}/{len(full_dataset)}\tuncertainty: {sample_uncer:.4f}",
                end="",
            )

    print()
    print("++++++++++++++++++++++++++++++++++++++++")
    # sort with ucnertainty
    results_uncertainty = torch.tensor(results_uncertainty)
    sorted_uncer, indices = torch.sort(results_uncertainty)
    print(f"uncer mean all: {torch.nanmean(results_uncertainty)}")
    subset_indices = indices[: int(indices.shape[0] * mining_portation)]
    print(f"uncer mean mining: {torch.nanmean(results_uncertainty[subset_indices])}")
    subset = data.Subset(org_dataset, indices=subset_indices)
    print(f"subset length: {len(subset)}/{len(org_dataset)}")
    return subset


def generator_mining_steps(
    total_epoch: int,
    start_portation: float,
    end_portation,
    re_mining_period: int,
) -> tuple[bool, float, int]:
    # mining_epochs = sorted(mining_epochs)
    for i in range(total_epoch):
        portation = max(
            start_portation + i * (end_portation - start_portation) / total_epoch, 1
        )
        if i % re_mining_period == 0:
            yield (True, portation, i)
        else:
            yield (False, portation, i)


def generator_mining_list(
    total_epoch: int, re_mining_epochs: list[int], re_mining_portions: list[float]
) -> tuple[bool, Optional[float], int]:
    for i in range(total_epoch):
        try:
            index = re_mining_epochs.index(i)
            yield (True, min(re_mining_portions[index], 1), i)
        except ValueError:
            yield (False, None, i)


def train_one_epoch(
    model,
    optimizer,
    train_loader_org,
    train_loader_tar,
    model_func,
    lr_scheduler,
    accumulated_iter,
    optim_cfg,
    rank,
    tbar,
    total_it_each_epoch,
    tb_log=None,
    leave_pbar=False,
    epoch=0,
    n_epoch=20,
    mt_head=None,
    with_source=True,
):
    """

    :param model:
    :param optimizer:
    :param train_loader:
    :param model_func: takes `model` and `batch` as input, output `loss`,
    `tb_dict`, `disp_dict`, where `tb_dict` include expected log scalars.
    :param lr_scheduler:
    :param accumulated_iter:
    :param optim_cfg:
    :param rank:
    :param tbar:
    :param total_it_each_epoch:
    :param dataloader_iter:
    :param tb_log:
    :param leave_pbar:
    :return:
    """

    loader_iter_org = iter(train_loader_org)
    loader_iter_tar = iter(train_loader_tar)

    if rank == 0:
        pbar = tqdm.tqdm(
            total=total_it_each_epoch,
            leave=leave_pbar,
            desc="train",
            dynamic_ncols=True,
        )

    essential_module = (
        model.module
        if isinstance(model, nn.parallel.DistributedDataParallel)
        else model
    )
    if hasattr(essential_module, "split_parameters"):
        point_params, image_params = essential_module.split_parameters()

    else:
        point_params = model.parameters()
    model.train()
    for cur_it in range(total_it_each_epoch):
        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except Exception:
            cur_lr = optimizer.param_groups[0]["lr"]

        if tb_log is not None:
            tb_log.add_scalar("meta_data/learning_rate", cur_lr, accumulated_iter)

        optimizer.zero_grad()
        p = float(cur_it + epoch * total_it_each_epoch) / float(
            n_epoch * total_it_each_epoch
        )
        # weight = 1e-4
        if epoch < 1:
            weight = 0.1
        else:
            weight = 1.0  # weight = 1.0
        alpha = weight * 2.0 / (1.0 + math.exp(-10 * p)) - 1
        disp_dict = {}
        tb_dict = {}
        if with_source:
            # source domain
            try:
                batch = next(loader_iter_org)
            except StopIteration:
                loader_iter_org = iter(train_loader_org)
                batch = next(loader_iter_org)
            batch["domain_label"] = 0
            batch["alpha"] = alpha
            if mt_head is not None:
                batch["mt_predict_branch"] = mt_head
            loss_org, tb_dict, disp_dict, _ = model_func(model, batch)
            loss_org.backward()
        else:
            loss_org = torch.tensor([0.0])

        try:
            batch = next(loader_iter_tar)
        except StopIteration:
            loader_iter_tar = iter(train_loader_tar)
            batch = next(loader_iter_tar)
        batch["domain_label"] = 1
        batch["alpha"] = alpha
        if mt_head is not None:
            batch["mt_predict_branch"] = mt_head
        loss_tar, tb_dict2, disp_dict2, _ = model_func(model, batch)
        tb_dict.update(tb_dict2)
        loss_tar.backward()
        loss = loss_tar.item() + loss_org.item()

        clip_grad_norm_(point_params, optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()
        if mt_head is not None:  # update
            essential_module.roi_head.update_mt_branch(mt_head)

        accumulated_iter += 1
        disp_dict.update(
            {
                "loss_det": disp_dict.get("loss_det", "none"),
                "loss_al": disp_dict2.get("loss_al", "none"),
                "loss_cl": disp_dict2.get("loss_cl", "none"),
                "loss_phst": disp_dict2.get("loss_phst", "none"),
                "lr": cur_lr,
            }
        )

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar("train/loss", loss, accumulated_iter)
                tb_log.add_scalar("meta_data/learning_rate", cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar("train/" + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(
    model,
    optimizer,
    train_loader_org,
    train_loader_tar: data.DataLoader,
    model_func,
    lr_scheduler,
    optim_cfg,
    start_epoch,
    total_epochs,
    start_iter,
    rank,
    tb_log,
    ckpt_save_dir,
    train_sampler_org=None,
    train_sampler_tar=None,
    lr_warmup_scheduler=None,
    ckpt_save_interval=3,
    max_ckpt_save_num=50,
    merge_all_iters_to_one_epoch=False,
    logger=None,
    with_source=True,
    with_uncer_mining=True,
    mining_schedule_generator=None,
    random_experiment_flag=False,
    train_dataset_tar_full=None,
):
    with_uncer_mining = with_uncer_mining and mining_schedule_generator is not None
    if mining_schedule_generator is not None:
        mining_scheduler = iter(mining_schedule_generator)

    accumulated_iter = start_iter
    pure_model = (
        model.module
        if isinstance(model, nn.parallel.DistributedDataParallel)
        else model
    )
    mt_head = None
    if hasattr(pure_model.roi_head, "init_mt_branch"):
        mt_head = pure_model.roi_head.init_mt_branch()
        if logger is not None:
            logger.info("Use mean teach predict head!")

    with tqdm.trange(
        start_epoch, total_epochs, desc="epochs", dynamic_ncols=True, leave=(rank == 0)
    ) as tbar:
        if merge_all_iters_to_one_epoch:
            raise NotImplementedError
            assert hasattr(train_loader_org.dataset, "merge_all_iters_to_one_epoch")
            train_loader_org.dataset.merge_all_iters_to_one_epoch(
                merge=True, epochs=total_epochs
            )

            assert hasattr(train_loader_tar.dataset, "merge_all_iters_to_one_epoch")
            train_loader_tar.dataset.merge_all_iters_to_one_epoch(
                merge=True, epochs=total_epochs
            )
            total_it_each_epoch = len(train_loader_org) // max(total_epochs, 1)
        for cur_epoch in tbar:
            if train_sampler_org is not None:
                train_sampler_org.set_epoch(cur_epoch)
            if train_sampler_tar is not None:
                train_sampler_tar.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            need_mining, mining_portation, _ = next(mining_scheduler)
            if with_uncer_mining and need_mining:
                print("-" * 30)
                print(
                    f"start mining at epoch {cur_epoch}, mining poration: {mining_portation}"
                )
                start_time = time.perf_counter()
                mining_set = uncertainty_mining(
                    model,
                    train_dataset_tar_full,
                    org_loader=train_loader_tar,
                    mining_portation=mining_portation,
                    random_choices_flag=random_experiment_flag,
                )
                print(f"mining time cost: {time.perf_counter() - start_time}")
                train_loader_tar = data.DataLoader(
                    mining_set,
                    batch_size=train_loader_tar.batch_size,
                    pin_memory=True,
                    num_workers=train_loader_tar.num_workers,
                    shuffle=(train_loader_tar.sampler is None),
                    collate_fn=train_loader_tar.collate_fn,
                    drop_last=False,
                    timeout=0,
                )

            total_it_each_epoch = len(train_loader_tar)
            accumulated_iter = train_one_epoch(
                model,
                optimizer,
                train_loader_org,
                train_loader_tar,
                model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter,
                optim_cfg=optim_cfg,
                rank=rank,
                tbar=tbar,
                tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                epoch=cur_epoch,
                mt_head=mt_head,
                with_source=with_source,
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:
                ckpt_list = glob.glob(str(ckpt_save_dir / "checkpoint_epoch_*.pth"))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(
                        0, len(ckpt_list) - max_ckpt_save_num + 1
                    ):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ("checkpoint_epoch_%d" % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter),
                    filename=ckpt_name,
                )


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file", type=str, default=None, help="specify the config for training"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        required=False,
        help="batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=70,
        required=False,
        help="number of epochs to train for",
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="number of workers for dataloader"
    )
    parser.add_argument(
        "--extra_tag", type=str, default="default", help="extra tag for this experiment"
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="checkpoint to start from"
    )
    parser.add_argument(
        "--pretrained_model", type=str, default=None, help="pretrained_model"
    )
    parser.add_argument(
        "--launcher", choices=["none", "pytorch", "slurm"], default="none"
    )
    parser.add_argument(
        "--tcp_port", type=int, default=18888, help="tcp port for distrbuted training"
    )
    parser.add_argument(
        "--sync_bn", action="store_true", default=False, help="whether to use sync bn"
    )
    parser.add_argument("--fix_random_seed", default=True, help="")
    parser.add_argument(
        "--ckpt_save_interval", type=int, default=1, help="number of training epochs"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--max_ckpt_save_num",
        type=int,
        default=9999,  # save all because we need all eval results
        help="max number of saved checkpoint",
    )
    parser.add_argument(
        "--merge_all_iters_to_one_epoch", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        default=None,
        nargs=argparse.REMAINDER,
        help="set extra config keys if needed",
    )
    parser.add_argument(
        "--mining_start",
        type=float,
        default=0.2,
        help="start portation for mining",
    )
    parser.add_argument(
        "--mining_end",
        type=float,
        default=0.9,
        help="end portation for mining",
    )
    parser.add_argument(
        "--mining_period",
        type=int,
        default=10,
        help="epoch period for mining",
    )
    parser.add_argument("--mining_at", nargs="+", default=None, help="mining at epoch")
    parser.add_argument(
        "--mining_portion", nargs="+", default=None, help="mining portion at epoch"
    )

    parser.add_argument(
        "--without_mining",
        action="store_true",
        default=False,
        help="without instance weighting",
    )
    parser.add_argument(
        "--random_exp",
        action="store_true",
        default=False,
        help="random choice experiment",
    )

    parser.add_argument(
        "--max_waiting_mins", type=int, default=0, help="max waiting minutes"
    )
    parser.add_argument("--start_epoch", type=int, default=0, help="")
    parser.add_argument("--save_to_file", action="store_true", default=False, help="")
    parser.add_argument(
        "--without_source",
        action="store_true",
        default=False,
        help="whether use source dataset when self training",
    )

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = "/".join(
        args.cfg_file.split("/")[1:-1]
    )  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    if args.launcher == "none":
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(
            common_utils, "init_dist_%s" % args.launcher
        )(args.tcp_port, args.local_rank, backend="nccl")
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert (
            args.batch_size % total_gpus == 0
        ), "Batch size should match the number of gpus"
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    output_dir = cfg.ROOT_DIR / "output" / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / "ckpt"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / (
        "log_train_%s.txt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info("**********************Start logging**********************")
    gpu_list = (
        os.environ["CUDA_VISIBLE_DEVICES"]
        if "CUDA_VISIBLE_DEVICES" in os.environ.keys()
        else "ALL"
    )
    logger.info("CUDA_VISIBLE_DEVICES=%s" % gpu_list)

    if dist_train:
        logger.info("total_batch_size: %d" % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system("cp %s %s" % (args.cfg_file, output_dir))

    tb_log = (
        SummaryWriter(log_dir=str(output_dir / "tensorboard"))
        if cfg.LOCAL_RANK == 0
        else None
    )
    if args.random_exp:
        logger.warning("**********************Now Random Choice**********************")
    # -----------------------create dataloader & network & optimizer--------------------
    train_set_org, train_loader_org, train_sampler_org = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG_ORG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train,
        workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
    )
    train_set_tar, train_loader_tar, train_sampler_tar = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG_TAR,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train,
        workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        domain="t",
    )

    model = build_network(
        model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set_org
    )

    # freeze image branch
    if model.model_cfg.get("FREEZE_IMAGE_BRANCH"):
        _, image_params = model.split_parameters()
        for param in image_params:
            param.requires_grad_(False)

    model.cuda()
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(
            filename=args.pretrained_model, to_cpu=dist, logger=logger
        )
    elif hasattr(model, "load_params_for_img_branch"):
        model.load_params_for_img_branch(logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(
            args.ckpt, to_cpu=dist, optimizer=optimizer, logger=logger
        )
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / "*checkpoint_epoch_*.pth"))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=dist, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model.train()
    if dist_train:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()]
        )
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer,
        total_iters_each_epoch=len(train_loader_org),
        total_epochs=args.epochs,
        last_epoch=last_epoch,
        optim_cfg=cfg.OPTIMIZATION,
    )

    # -------------------- MINING related ------------------------------------
    if args.mining_at is not None and args.mining_portion is not None:
        re_mining_epochs = [int(epoch) for epoch in args.mining_at]
        re_mining_portions = [float(epoch) for epoch in args.mining_portion]
        assert max(re_mining_epochs) < args.epochs
        assert len(re_mining_epochs) == len(re_mining_portions)
        assert 0 in re_mining_epochs
        mining_schedule_generator = generator_mining_list(
            args.epochs, re_mining_epochs, re_mining_portions
        )
    else:
        mining_schedule_generator = generator_mining_steps(
            args.epochs,
            start_portation=args.mining_start,
            end_portation=args.mining_end,
            re_mining_period=args.mining_period,
        )
    mining_schedule_generator, mining_schedule_generator2 = tee(
        mining_schedule_generator
    )
    for i, j, k in list(iter(mining_schedule_generator2)):
        if i:
            print(f"start from epoch:{k} portion will be: {j}")

    # -----------------------start training---------------------------
    logger.info(
        "**********************Start training %s/%s(%s)**********************"
        % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag)
    )

    print("START EPOCH %d " % start_epoch)
    train_model(
        model,
        optimizer,
        train_loader_org,
        train_loader_tar,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        with_source=not args.without_source,
        with_uncer_mining=not args.without_mining,
        random_experiment_flag=args.random_exp,
        mining_schedule_generator=mining_schedule_generator,
        train_dataset_tar_full=train_set_tar,  # used in mining
    )

    logger.info(
        "**********************End training %s/%s(%s)**********************\n\n\n"
        % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag)
    )

    logger.info(
        "**********************Start evaluation %s/%s(%s)**********************"
        % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag)
    )
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG_TAR,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train,
        workers=args.workers,
        logger=logger,
        training=False,
    )
    eval_output_dir = output_dir / "eval" / "eval_with_train"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    # args.start_epoch = max(args.epochs - 30, 0)  # Only evaluate the last 10 epochs

    # we need full evaluation information
    args.start_epoch = 0
    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader,
        args,
        eval_output_dir,
        logger,
        ckpt_dir,
        dist_test=dist_train,
    )
    logger.info(
        "**********************End evaluation %s/%s(%s)**********************"
        % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag)
    )


if __name__ == "__main__":
    main()
