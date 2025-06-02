"""
Simplified RunnerBaseW for non-distributed 2-GPU training using DataParallel
"""

import datetime
import json
import logging
import os
import time
from pathlib import Path
from lavis.common.logger import MetricLogger, SmoothedValue
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ChainDataset
from lavis.datasets.data_utils import prepare_sample
import webdataset as wds
from torch import autocast
from lavis.common.dist_utils import (
    download_cached_file,
    is_main_process,
    main_process,
)
from lavis.common.registry import registry
from lavis.common.utils import is_url
from lavis.datasets.data_utils import concat_datasets
from lavis.datasets.dataloader_utils import IterLoader, MultiIterLoader, PrefetchLoader_


class RunnerBaseW:
    def __init__(self, cfg, task, model, datasets, job_id):
        self.config = cfg
        self.datasets = datasets
        self._model = model
        self.job_id = job_id

        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._scaler = None
        self._dataloaders = None
        self._lr_sched = None

        self.start_epoch = 0
        self.setup_output_dir()

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(self.config.run_cfg.device)
        return self._device

    @property
    def model(self):
        if self._model.device != self.device:
            self._model = self._model.to(self.device)
            if self._wrapped_model is None:
                self._wrapped_model = nn.DataParallel(self._model).cuda()
        return self._wrapped_model

    @property
    def optimizer(self):
        if self._optimizer is None:
            lr_scale = self.config.run_cfg.get("lr_layer_decay", 1)
            weight_decay = self.config.run_cfg.get("weight_decay", 0.05)
            optim_params = self._model.get_optimizer_params(weight_decay, lr_scale)

            beta2 = self.config.run_cfg.get("beta2", 0.999)
            self._optimizer = torch.optim.AdamW(
                optim_params,
                lr=float(self.config.run_cfg.init_lr),
                betas=(0.9, beta2),
            )
        return self._optimizer

    @property
    def scaler(self):
        if self.config.run_cfg.get("amp", False):
            if self._scaler is None:
                self._scaler = torch.cuda.amp.GradScaler()
        return self._scaler

    @property
    def lr_scheduler(self):
        if self._lr_sched is None:
            lr_sched_cls = registry.get_lr_scheduler_class(self.config.run_cfg.lr_sched)
            self._lr_sched = lr_sched_cls(
                optimizer=self.optimizer,
                max_epoch=self.max_epoch,
                min_lr=self.min_lr,
                init_lr=self.init_lr,
                decay_rate=self.config.run_cfg.get("lr_decay_rate", None),
                warmup_start_lr=self.config.run_cfg.get("warmup_lr", -1),
                warmup_steps=self.config.run_cfg.get("warmup_steps", 0),
            )
        return self._lr_sched

    @property
    def dataloaders(self):
        if self._dataloaders is None:
            datasets = [self.datasets[split] for split in sorted(self.datasets)]
            is_trains = [split in self.train_splits for split in sorted(self.datasets)]
            batch_sizes = [
                self.config.run_cfg.batch_size_train if is_train else self.config.run_cfg.batch_size_eval
                for is_train in is_trains
            ]
            collate_fns = [
                [getattr(d, "collater", None) for d in dataset] if isinstance(dataset, (list, tuple))
                else getattr(dataset, "collater", None)
                for dataset in datasets
            ]

            dataloaders = self.create_loaders(
                datasets,
                self.config.run_cfg.num_workers,
                batch_sizes,
                is_trains,
                collate_fns,
            )
            self._dataloaders = {k: v for k, v in zip(sorted(self.datasets), dataloaders)}
        return self._dataloaders

    def create_loaders(self, datasets, num_workers, batch_sizes, is_trains, collate_fns, dataset_ratios=None):
        def _create_loader(dataset, num_workers, bsz, is_train, collate_fn):
            if isinstance(dataset, ChainDataset) or isinstance(dataset, wds.DataPipeline):
                loader = iter(DataLoader(dataset, batch_size=bsz, num_workers=num_workers, pin_memory=True))
            else:
                loader = DataLoader(
                    dataset,
                    batch_size=bsz,
                    num_workers=num_workers,
                    pin_memory=True,
                    shuffle=is_train,
                    collate_fn=collate_fn,
                    drop_last=is_train,
                )
                loader = PrefetchLoader_(loader)
                if is_train:
                    loader = IterLoader(loader, use_distributed=False)
            return loader

        loaders = []
        for dataset, bsz, is_train, collate_fn in zip(datasets, batch_sizes, is_trains, collate_fns):
            if isinstance(dataset, (list, tuple)):
                loader = MultiIterLoader(
                    loaders=[
                        _create_loader(d, num_workers, bsz, is_train, collate_fn[i])
                        for i, d in enumerate(dataset)
                    ],
                    ratios=dataset_ratios,
                )
            else:
                loader = _create_loader(dataset, num_workers, bsz, is_train, collate_fn)
            loaders.append(loader)
        return loaders

    @property
    def train_loader(self):
        return self.dataloaders["train"]

    def train(self):
        best_val_loss_so_far = float("inf")
        best_epoch = 0
        self.log_config()

        if self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            if not self.evaluate_only:
                train_stats = self.train_epoch(cur_epoch)
                self.log_stats(train_stats, "train")

            if len(self.valid_splits) > 0 and (self.evaluate_only or cur_epoch % self.val_freq == 0):
                for split_name in self.valid_splits:
                    val_log = self.eval_epoch(split_name, cur_epoch)
                    if val_log and is_main_process():
                        current_val_loss = val_log["loss"]
                        if current_val_loss < best_val_loss_so_far and split_name == "val":
                            best_epoch, best_val_loss_so_far = cur_epoch, current_val_loss
                            if not self.evaluate_only:
                                self._save_checkpoint(cur_epoch, is_best=True)
                        val_log.update({"best_epoch": best_epoch})
                        self.log_stats(val_log, split_name)

            if self.evaluate_only:
                break

        if not os.path.exists(os.path.join(self.output_dir, "checkpoint_best.pth")):
            self._save_checkpoint(cur_epoch, is_best=True)


    def train_epoch(self, epoch):
        self.model.train()

        stats = self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=len(self.train_loader),
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            log_freq=self.log_freq,
            cuda_enabled=self.cuda_enabled,
            accum_grad_iters=self.accum_grad_iters,
        )

        stats = {f"{k}": v for k, v in stats.items()}
        stats["epoch"] = epoch
        return stats




    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, skip_reload=False):
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, f"DataLoader for split {split_name} is None."
        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        self.before_evaluation(model=model, dataset=self.datasets[split_name])
        results = self.evaluation(self.config, model, data_loader)

        if results is not None:
            return self.after_evaluation(self.config, results, split_name, cur_epoch)
        return None

    def unwrap_dist_model(self, model):
        return model.module if isinstance(model, nn.DataParallel) else model

    @main_process
    def _save_checkpoint(self, cur_epoch, is_best=False):
        model_no_ddp = self.unwrap_dist_model(self.model)
        state_dict = {
            k: v for k, v in model_no_ddp.state_dict().items() if v.requires_grad
        }
        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }
        path = os.path.join(self.output_dir, f"checkpoint_{'best' if is_best else cur_epoch}.pth")
        logging.info(f"Saving checkpoint at epoch {cur_epoch} to {path}.")
        torch.save(save_obj, path)

    def _reload_best_model(self, model):
        checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")
        logging.info(f"Loading checkpoint from {checkpoint_path}.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        try:
            model.load_state_dict(checkpoint["model"])
        except:
            model.load_state_dict(checkpoint["model"], strict=False)
        return model

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.unwrap_dist_model(self.model).load_state_dict(checkpoint["model"], strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.start_epoch = checkpoint["epoch"] + 1

    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {f"{split_name}_{k}": v for k, v in stats.items()}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")

    def setup_output_dir(self):
        lib_root = Path(registry.get_path("library_root"))
        output_dir = lib_root / self.config.run_cfg.output_dir / self.job_id
        result_dir = output_dir / "result"
        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        registry.register_path("result_dir", str(result_dir))
        registry.register_path("output_dir", str(output_dir))
        self.result_dir = result_dir
        self.output_dir = output_dir

    # Other properties remain the same as yours
    @property
    def cuda_enabled(self): return self.device.type == "cuda"
    @property
    def max_epoch(self): return int(self.config.run_cfg.max_epoch)
    @property
    def log_freq(self): return int(self.config.run_cfg.get("log_freq", 50))
    @property
    def save_freq(self): return int(self.config.run_cfg.get("save_freq", 5))
    @property
    def val_freq(self): return int(self.config.run_cfg.get("val_freq", 1))
    @property
    def save_last(self): return int(self.config.run_cfg.get("save_last", True))
    @property
    def init_lr(self): return float(self.config.run_cfg.init_lr)
    @property
    def min_lr(self): return float(self.config.run_cfg.min_lr)
    @property
    def accum_grad_iters(self): return int(self.config.run_cfg.get("accum_grad_iters", 1))
    @property
    def valid_splits(self): return self.config.run_cfg.get("valid_splits", [])
    @property
    def test_splits(self): return self.config.run_cfg.get("test_splits", [])
    @property
    def train_splits(self): return self.config.run_cfg.get("train_splits", [])
    @property
    def evaluate_only(self): return self.config.run_cfg.evaluate
    @property
    def resume_ckpt_path(self): return self.config.run_cfg.get("resume_ckpt_path", None)
    @property
    def use_dist_eval_sampler(self): return False






    def build_model(self, cfg):
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        datasets = dict()
        datasets_config = cfg.datasets_cfg
        # print("\n>>> [DEBUG] datasets_cfg-------------------------:", cfg.datasets_cfg)
        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]
            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()
            datasets[name] = dataset

        # print(">>> [DEBUG] Built datasets keys:", datasets.keys())
        return datasets

    # def train_step(self, model, samples):
    #     output = model(samples)
    #     loss_dict = {k: v for k, v in output.items() if "loss" in k}
    #     return output["loss"], loss_dict


    def train_step(self, model, samples):
        output = model(samples)
        raw_loss = output["loss"]

        # Ensure it's a scalar (e.g. mean reduction)
        if raw_loss.ndim != 0:
            raw_loss = raw_loss.mean()

        loss_dict = {k: v.mean().item() if v.ndim > 0 else v.item() for k, v in output.items() if "loss" in k}
        return raw_loss, loss_dict



    def valid_step(self,config, model, samples):
        model.eval()
        samples = prepare_sample(samples, cuda_enabled=True)

        retrieval_eval = getattr(config.run_cfg, "retrieval_eval", False)
        use_amp = True

        with torch.no_grad():
            with autocast("cuda", enabled=use_amp):
                if retrieval_eval:
                    # Only extract embeddings for retrieval
                    features = model.forward_features(samples)
                    rgb_feats = features["rgb_feats"]
                    lidar_feats = features["lidar_feats"]
                    # print("[DEBUG] rgb_feats type:", type(rgb_feats), "shape:", rgb_feats.shape if isinstance(rgb_feats, torch.Tensor) else "N/A")
                    # print("[DEBUG] lidar_feats type:", type(lidar_feats), "shape:", lidar_feats.shape if isinstance(lidar_feats, torch.Tensor) else "N/A")

                    return {"loss": None, "output": {"rgb_feats": rgb_feats, "lidar_feats": lidar_feats}}
                else:
                    # Use standard forward that returns loss
                    output = model(samples)
                    loss = output["loss"].item() if "loss" in output else None
                    return {"loss": loss, "output": output}


    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))



    def after_evaluation(self, config, val_result, split_name, epoch):
        """
        Process the results after evaluation.

        Args:
            val_result (dict): a dict containing evaluation outputs (e.g., losses, metrics, retrieval stats).
            split_name (str): name of the split evaluated.
            epoch (int or str): epoch number or "best".

        Returns:
            dict: stats to log (should contain "loss" key at least, or retrieval metrics).
        """
        print("[AFTER_EVAL] Received val_result:", val_result)

        stats = {
            "epoch": epoch,
            "split": split_name,
        }

        if isinstance(val_result, dict):
            # Log loss if available
            if "loss" in val_result:
                stats["loss"] = val_result["loss"]

            # Log general metrics if present
            if "metrics" in val_result:
                stats.update(val_result["metrics"])

            # If retrieval evaluation results were added
            if "output" in val_result and config.run_cfg.get("retrieval_eval", False):
                retrieval_metrics = self.compute_retrieval_metrics_from_outputs(val_result["output"])
                stats.update(retrieval_metrics)

            if "retrieval" in val_result:
                retrieval_metrics = val_result["retrieval"]
                stats.update(retrieval_metrics)

        print("[AFTER_EVAL] Returning stats:", stats)
        return stats

    @torch.no_grad()
    def evaluation(self, config, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        print_freq = 10

        total_loss = 0.0
        num_batches = 0
        all_outputs = []

        retrieval_eval = getattr(config.run_cfg, "retrieval_eval", False)

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            eval_output = self.valid_step(config, model=model, samples=samples)

            if retrieval_eval:
                all_outputs.append(eval_output)  # raw features
            else:
                loss = eval_output.get("loss", None)
                output = eval_output.get("output", None)
                if loss is not None:
                    total_loss += loss
                all_outputs.append(output)

            num_batches += 1

        if retrieval_eval:
            return {
                "output": all_outputs  # for computing Recall@K etc.
            }
        else:
            avg_loss = total_loss / max(num_batches, 1)
            return {
                "loss": avg_loss,
                "output": all_outputs,
            }




    def compute_retrieval_metrics_from_outputs(self, output_list, top_k=(1, 5, 10)):
        rgb_feats_list = []
        lidar_feats_list = []

        for out in output_list:
            inner_output = out["output"]
            rgb_feats_list.append(inner_output["rgb_feats"])
            lidar_feats_list.append(inner_output["lidar_feats"])

        rgb_feats = torch.cat(rgb_feats_list, dim=0)      # [B_total, N, D]
        lidar_feats = torch.cat(lidar_feats_list, dim=0)  # [B_total, N, D]

        rgb_flat = rgb_feats.mean(dim=1)      # [B_total, D]
        lidar_flat = lidar_feats.mean(dim=1)  # [B_total, D]

        sim_matrix = torch.matmul(rgb_flat, lidar_flat.T)  # [B, B]

        metrics = {}

        # === Image-to-LiDAR ===
        B = sim_matrix.size(0)
        for k in top_k:
            hits = sum([i in torch.topk(sim_matrix[i], k=k).indices for i in range(B)])
            metrics[f"recall@{k}_rgb2lidar"] = hits / B

        ranks = [(torch.argsort(sim_matrix[i], descending=True) == i).nonzero(as_tuple=True)[0].item() + 1 for i in range(B)]
        metrics["mrr_rgb2lidar"] = sum(1.0 / rank for rank in ranks) / B

        # === LiDAR-to-Image ===
        sim_matrix_T = sim_matrix.T  # [B, B]
        for k in top_k:
            hits = sum([i in torch.topk(sim_matrix_T[i], k=k).indices for i in range(B)])
            metrics[f"recall@{k}_lidar2rgb"] = hits / B

        ranks = [(torch.argsort(sim_matrix_T[i], descending=True) == i).nonzero(as_tuple=True)[0].item() + 1 for i in range(B)]
        metrics["mrr_lidar2rgb"] = sum(1.0 / rank for rank in ranks) / B

        return metrics






    


    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        logging.info(f"Start training epoch {epoch}, {iters_per_epoch} iters per inner epoch.")
        header = f"Train: data epoch: [{epoch}]"
        inner_epoch = epoch if start_iters is None else start_iters // iters_per_epoch
        if start_iters is not None:
            header += f"; inner epoch [{inner_epoch}]"
        
        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            if samples is None or samples.get("is_empty", False):
                continue
            
            if not isinstance(samples, dict):
                samples = {"is_empty": True}

            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)
            print(f"[DEBUG] Step {i}, LR: {optimizer.param_groups[0]['lr']:.8f} -----------------------")


            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, loss_dict = self.train_step(model=model, samples=samples)
                loss /= accum_grad_iters

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(**loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            # k: "{:.3f}".format(meter.global_avg)
            k: ("{:.8f}".format(meter.global_avg) if k == "lr" else "{:.3f}".format(meter.global_avg))
            for k, meter in metric_logger.meters.items()
        }





# def setup_task(cfg):
#     assert "task" in cfg.run_cfg, "Task name must be provided."
#     task_name = cfg.run_cfg.task
#     # task = registry.get_task_class(task_name).setup_task(cfg=cfg)
#     task = BaseTask()
#     assert task is not None, f"Task {task_name} not properly registered."
#     return task














# import logging
# import os
# from torch import autocast
# import torch

# from lavis.common.logger import MetricLogger, SmoothedValue
# from lavis.common.registry import registry
# from lavis.datasets.data_utils import prepare_sample


# class BaseTask:
#     def __init__(self, **kwargs):
#         super().__init__()
#         self.inst_id_key = "instance_id"

#     @classmethod
#     def setup_task(cls, **kwargs):
#         return cls()

#     def build_model(self, cfg):
#         model_config = cfg.model_cfg
#         model_cls = registry.get_model_class(model_config.arch)
#         return model_cls.from_config(model_config)

#     def build_datasets(self, cfg):
#         datasets = dict()
#         datasets_config = cfg.datasets_cfg
#         # print("\n>>> [DEBUG] datasets_cfg-------------------------:", cfg.datasets_cfg)
#         assert len(datasets_config) > 0, "At least one dataset has to be specified."

#         for name in datasets_config:
#             dataset_config = datasets_config[name]
#             builder = registry.get_builder_class(name)(dataset_config)
#             dataset = builder.build_datasets()
#             datasets[name] = dataset

#         # print(">>> [DEBUG] Built datasets keys:", datasets.keys())
#         return datasets

#     # def train_step(self, model, samples):
#     #     output = model(samples)
#     #     loss_dict = {k: v for k, v in output.items() if "loss" in k}
#     #     return output["loss"], loss_dict


#     def train_step(self, model, samples):
#         output = model(samples)
#         raw_loss = output["loss"]

#         # Ensure it's a scalar (e.g. mean reduction)
#         if raw_loss.ndim != 0:
#             raw_loss = raw_loss.mean()

#         loss_dict = {k: v.mean().item() if v.ndim > 0 else v.item() for k, v in output.items() if "loss" in k}
#         return raw_loss, loss_dict



#     def valid_step(self,config, model, samples):
#         model.eval()
#         samples = prepare_sample(samples, cuda_enabled=True)

#         retrieval_eval = getattr(config.run_cfg, "retrieval_eval", False)
#         use_amp = True

#         with torch.no_grad():
#             with autocast("cuda", enabled=use_amp):
#                 if retrieval_eval:
#                     # Only extract embeddings for retrieval
#                     features = model.forward_features(samples)
#                     rgb_feats = features["rgb_feats"]
#                     lidar_feats = features["lidar_feats"]
#                     # print("[DEBUG] rgb_feats type:", type(rgb_feats), "shape:", rgb_feats.shape if isinstance(rgb_feats, torch.Tensor) else "N/A")
#                     # print("[DEBUG] lidar_feats type:", type(lidar_feats), "shape:", lidar_feats.shape if isinstance(lidar_feats, torch.Tensor) else "N/A")

#                     return {"loss": None, "output": {"rgb_feats": rgb_feats, "lidar_feats": lidar_feats}}
#                 else:
#                     # Use standard forward that returns loss
#                     output = model(samples)
#                     loss = output["loss"].item() if "loss" in output else None
#                     return {"loss": loss, "output": output}


#     def before_evaluation(self, model, dataset, **kwargs):
#         model.before_evaluation(dataset=dataset, task_type=type(self))



#     def after_evaluation(self, config, val_result, split_name, epoch):
#         """
#         Process the results after evaluation.

#         Args:
#             val_result (dict): a dict containing evaluation outputs (e.g., losses, metrics, retrieval stats).
#             split_name (str): name of the split evaluated.
#             epoch (int or str): epoch number or "best".

#         Returns:
#             dict: stats to log (should contain "loss" key at least, or retrieval metrics).
#         """
#         print("[AFTER_EVAL] Received val_result:", val_result)

#         stats = {
#             "epoch": epoch,
#             "split": split_name,
#         }

#         if isinstance(val_result, dict):
#             # Log loss if available
#             if "loss" in val_result:
#                 stats["loss"] = val_result["loss"]

#             # Log general metrics if present
#             if "metrics" in val_result:
#                 stats.update(val_result["metrics"])

#             # If retrieval evaluation results were added
#             if "output" in val_result and config.run_cfg.get("retrieval_eval", False):
#                 retrieval_metrics = self.compute_retrieval_metrics_from_outputs(val_result["output"])
#                 stats.update(retrieval_metrics)

#             if "retrieval" in val_result:
#                 retrieval_metrics = val_result["retrieval"]
#                 stats.update(retrieval_metrics)

#         print("[AFTER_EVAL] Returning stats:", stats)
#         return stats

#     @torch.no_grad()
#     def evaluation(self, config, model, data_loader, cuda_enabled=True):
#         metric_logger = MetricLogger(delimiter="  ")
#         header = "Evaluation"
#         print_freq = 10

#         total_loss = 0.0
#         num_batches = 0
#         all_outputs = []

#         retrieval_eval = getattr(config.run_cfg, "retrieval_eval", False)

#         for samples in metric_logger.log_every(data_loader, print_freq, header):
#             samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
#             eval_output = self.valid_step(config, model=model, samples=samples)

#             if retrieval_eval:
#                 all_outputs.append(eval_output)  # raw features
#             else:
#                 loss = eval_output.get("loss", None)
#                 output = eval_output.get("output", None)
#                 if loss is not None:
#                     total_loss += loss
#                 all_outputs.append(output)

#             num_batches += 1

#         if retrieval_eval:
#             return {
#                 "output": all_outputs  # for computing Recall@K etc.
#             }
#         else:
#             avg_loss = total_loss / max(num_batches, 1)
#             return {
#                 "loss": avg_loss,
#                 "output": all_outputs,
#             }




#     def compute_retrieval_metrics_from_outputs(self, output_list, top_k=(1, 5, 10)):
#         rgb_feats_list = []
#         lidar_feats_list = []

#         for out in output_list:
#             inner_output = out["output"]
#             rgb_feats_list.append(inner_output["rgb_feats"])
#             lidar_feats_list.append(inner_output["lidar_feats"])

#         rgb_feats = torch.cat(rgb_feats_list, dim=0)      # [B_total, N, D]
#         lidar_feats = torch.cat(lidar_feats_list, dim=0)  # [B_total, N, D]

#         rgb_flat = rgb_feats.mean(dim=1)      # [B_total, D]
#         lidar_flat = lidar_feats.mean(dim=1)  # [B_total, D]

#         sim_matrix = torch.matmul(rgb_flat, lidar_flat.T)  # [B, B]

#         metrics = {}

#         # === Image-to-LiDAR ===
#         B = sim_matrix.size(0)
#         for k in top_k:
#             hits = sum([i in torch.topk(sim_matrix[i], k=k).indices for i in range(B)])
#             metrics[f"recall@{k}_rgb2lidar"] = hits / B

#         ranks = [(torch.argsort(sim_matrix[i], descending=True) == i).nonzero(as_tuple=True)[0].item() + 1 for i in range(B)]
#         metrics["mrr_rgb2lidar"] = sum(1.0 / rank for rank in ranks) / B

#         # === LiDAR-to-Image ===
#         sim_matrix_T = sim_matrix.T  # [B, B]
#         for k in top_k:
#             hits = sum([i in torch.topk(sim_matrix_T[i], k=k).indices for i in range(B)])
#             metrics[f"recall@{k}_lidar2rgb"] = hits / B

#         ranks = [(torch.argsort(sim_matrix_T[i], descending=True) == i).nonzero(as_tuple=True)[0].item() + 1 for i in range(B)]
#         metrics["mrr_lidar2rgb"] = sum(1.0 / rank for rank in ranks) / B

#         return metrics





#     def train_epoch(
#         self,
#         epoch,
#         model,
#         data_loader,
#         optimizer,
#         lr_scheduler,
#         scaler=None,
#         cuda_enabled=False,
#         log_freq=50,
#         accum_grad_iters=1,
#     ):
#         stats = self._train_inner_loop(
#             epoch=epoch,
#             iters_per_epoch=len(data_loader),
#             model=model,
#             data_loader=data_loader,
#             optimizer=optimizer,
#             scaler=scaler,
#             lr_scheduler=lr_scheduler,
#             log_freq=log_freq,
#             cuda_enabled=cuda_enabled,
#             accum_grad_iters=accum_grad_iters,
#         )
#         # Add epoch number to log
#         # stats["epoch"] = epoch
#         stats = {f"{k}": v for k, v in stats.items()}
#         stats["epoch"] = epoch
#         return stats
    


#     def _train_inner_loop(
#         self,
#         epoch,
#         iters_per_epoch,
#         model,
#         data_loader,
#         optimizer,
#         lr_scheduler,
#         scaler=None,
#         start_iters=None,
#         log_freq=50,
#         cuda_enabled=False,
#         accum_grad_iters=1,
#     ):
#         use_amp = scaler is not None

#         if not hasattr(data_loader, "__next__"):
#             data_loader = iter(data_loader)

#         metric_logger = MetricLogger(delimiter="  ")
#         metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
#         metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

#         logging.info(f"Start training epoch {epoch}, {iters_per_epoch} iters per inner epoch.")
#         header = f"Train: data epoch: [{epoch}]"
#         inner_epoch = epoch if start_iters is None else start_iters // iters_per_epoch
#         if start_iters is not None:
#             header += f"; inner epoch [{inner_epoch}]"
        
#         for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
#             if i >= iters_per_epoch:
#                 break

#             samples = next(data_loader)
#             samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
#             if samples is None or samples.get("is_empty", False):
#                 continue
            
#             if not isinstance(samples, dict):
#                 samples = {"is_empty": True}

#             samples.update(
#                 {
#                     "epoch": inner_epoch,
#                     "num_iters_per_epoch": iters_per_epoch,
#                     "iters": i,
#                 }
#             )

#             lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)
#             print(f"[DEBUG] Step {i}, LR: {optimizer.param_groups[0]['lr']:.8f} -----------------------")


#             with torch.cuda.amp.autocast(enabled=use_amp):
#                 loss, loss_dict = self.train_step(model=model, samples=samples)
#                 loss /= accum_grad_iters

#             if use_amp:
#                 scaler.scale(loss).backward()
#             else:
#                 loss.backward()

#             if (i + 1) % accum_grad_iters == 0:
#                 if use_amp:
#                     scaler.step(optimizer)
#                     scaler.update()
#                 else:
#                     optimizer.step()
#                 optimizer.zero_grad()

#             metric_logger.update(**loss_dict)
#             metric_logger.update(lr=optimizer.param_groups[0]["lr"])

#         metric_logger.synchronize_between_processes()
#         logging.info("Averaged stats: " + str(metric_logger.global_avg()))
#         return {
#             # k: "{:.3f}".format(meter.global_avg)
#             k: ("{:.8f}".format(meter.global_avg) if k == "lr" else "{:.3f}".format(meter.global_avg))
#             for k, meter in metric_logger.meters.items()
#         }


# ----------------------
# Simple Task Setup
# ----------------------



















