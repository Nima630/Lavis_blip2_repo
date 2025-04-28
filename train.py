"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

# import lavis.tasks as tasks
from lavis.tasks.base_task import setup_task

from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
# from lavis.tasks import *
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config): # ----------------------------------------------------------------------------------------------------------
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg): # ----------------------------------------------------------------------------------------------------------
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    print("[TRACE] Starting train.py")
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()
    #  Print all registered models
    # print("✅ Registered models:", registry.mapping["model_name_mapping"].keys())

    cfg = Config(parse_args())

    
    
    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    # task = tasks.setup_task(cfg)
    task = setup_task(cfg)
    datasets = task.build_datasets(cfg)
    # print(">>> [DEBUG] Dataset type for 'train':", type(datasets.get("train")))
    # print(">>> [DEBUG] Train dataset length:", len(datasets.get("train")))
    train_dataset = datasets.get("coco_caption", {}).get("train")
    if train_dataset is None:
        raise ValueError("Train dataset not found in 'coco_caption'.")
    else:
        print(">>> [DEBUG] Dataset type for 'train':", type(train_dataset))
        print(">>> [DEBUG] Train dataset length:", len(train_dataset))

    # if isinstance(datasets.get("train"), (tuple, list)):
    #     for i, d in enumerate(datasets["train"]):
    #         print(f">>> [DEBUG] train[{i}] type: {type(d)}")


    # Create a DataLoader to inspect samples
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    print("\n>>> [DEBUG] Sample inspection:")
    for idx, sample in enumerate(dataloader):
        print(f"Sample {idx}:")
        for key in sample:
            print(f"  {key}: {type(sample[key])}, shape: {sample[key].shape if hasattr(sample[key], 'shape') else sample[key]}")
        if idx >= 2:
            break

    model = task.build_model(cfg)

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    main()
