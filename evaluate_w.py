import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from lavis.datasets.waymo import WaymoDatasetBuilder
from lavis.models.blip2_models.blip2_qformer_w import Blip2Qformer  # Custom model
from lavis.tasks.base_task_w import BaseTask  # Custom task
from lavis.common.config_w import Config  # Custom config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.utils import now
from lavis.runners.runner_base_w import RunnerBaseW  # Custom runner
from lavis.common.registry import registry
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate")
    parser.add_argument("--cfg-path", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="Override options in YAML with key=value",
    )
    return parser.parse_args()

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def main():
    job_id = now()
    print("[TRACE] Starting evaluate_w.py")
    args = parse_args()
    cfg = Config(args)

    init_distributed_mode(cfg.run_cfg)
    cfg.run_cfg.gpu = int(os.environ.get("LOCAL_RANK", 0))
    setup_seeds(cfg)
    setup_logger()

    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lavis")
    registry.register_path("library_root", root_dir)
    repo_root = os.path.join(root_dir, "..")
    registry.register_path("repo_root", repo_root)
    registry.register("MAX_INT", sys.maxsize)
    registry.register("SPLIT_NAMES", ["train", "val", "test"])

    print(cfg.run_cfg)
    print("cfg.run_cfg")

    task = BaseTask()
    builder = WaymoDatasetBuilder(cfg)
    datasets = builder.build_datasets()

    model = Blip2Qformer.from_config(cfg.model_cfg)
    if not cfg.run_cfg.amp:
        model = model.float()

    runner = RunnerBaseW(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)
    runner.evaluate(skip_reload=True)

if __name__ == "__main__":
    main()
































# """
#  Copyright (c) 2022, salesforce.com, inc.
#  All rights reserved.
#  SPDX-License-Identifier: BSD-3-Clause
#  For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
# """

# import argparse
# import random

# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn

# import lavis.tasks as tasks
# from lavis.common.config import Config
# from lavis.common.dist_utils import get_rank, init_distributed_mode
# from lavis.common.logger import setup_logger
# from lavis.common.optims import (
#     LinearWarmupCosineLRScheduler,
#     LinearWarmupStepLRScheduler,
# )
# from lavis.common.utils import now

# # imports modules for registration
# from lavis.datasets.builders import *
# from lavis.models import *
# from lavis.processors import *
# from lavis.runners.runner_base import RunnerBase
# from lavis.tasks import *


# def parse_args():
#     parser = argparse.ArgumentParser(description="Training")

#     parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
#     parser.add_argument(
#         "--options",
#         nargs="+",
#         help="override some settings in the used config, the key-value pair "
#         "in xxx=yyy format will be merged into config file (deprecate), "
#         "change to --cfg-options instead.",
#     )

#     args = parser.parse_args()
#     # if 'LOCAL_RANK' not in os.environ:
#     #     os.environ['LOCAL_RANK'] = str(args.local_rank)

#     return args


# def setup_seeds(config):
#     seed = config.run_cfg.seed + get_rank()

#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     cudnn.benchmark = False
#     cudnn.deterministic = True


# def main():
#     # allow auto-dl completes on main process without timeout when using NCCL backend.
#     # os.environ["NCCL_BLOCKING_WAIT"] = "1"

#     # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
#     job_id = now()

#     cfg = Config(parse_args())

#     init_distributed_mode(cfg.run_cfg)

#     setup_seeds(cfg)

#     # set after init_distributed_mode() to only log on master.
#     setup_logger()

#     cfg.pretty_print()

#     task = tasks.setup_task(cfg)
#     datasets = task.build_datasets(cfg)
#     model = task.build_model(cfg)

#     runner = RunnerBase(
#         cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
#     )
#     runner.evaluate(skip_reload=True)


# if __name__ == "__main__":
#     main()
