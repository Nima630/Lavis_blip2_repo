import logging
import os
from torch import autocast
import torch
import torch.distributed as dist
from lavis.common.dist_utils import (
    get_rank,
    get_world_size,
    is_main_process,
    is_dist_avail_and_initialized,
)
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample


class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()
        self.inst_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

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

    def train_step(self, model, samples):
        output = model(samples)
        loss_dict = {k: v for k, v in output.items() if "loss" in k}
        return output["loss"], loss_dict

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


    # def valid_step(self, model, samples):
    #     model.eval()
    #     samples = prepare_sample(samples, cuda_enabled=True)

    #     use_amp = True  # or however you track AMP usage (usually from config)

    #     with torch.no_grad():
    #         with autocast("cuda", enabled=use_amp):
    #             output = model(samples)

    #     if "loss" in output:
    #         loss = output["loss"].item()
    #     else:
    #         loss = None

    #     print(f"[VALIDATION] Batch validation output: {output}")
    #     print(f"[VALIDATION] Batch validation loss: {loss}")

    #     return {"loss": loss, "output": output}


    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))


    # def after_evaluation(self, val_result, split_name, epoch):
    #     """
    #     Process the results after evaluation.
        
    #     Args:
    #         val_result (dict): a dict containing evaluation outputs (e.g., losses, metrics).
    #         split_name (str): name of the split evaluated.
    #         epoch (int): epoch number.
            
    #     Returns:
    #         dict: stats to log (should contain "loss" key at least).
    #     """
    #     print("[AFTER_EVAL] Received val_result:", val_result)
        
    #     stats = {}
    #     if isinstance(val_result, dict):
    #         if "loss" in val_result:
    #             stats["loss"] = val_result["loss"]
    #         # Optionally, include more metrics if available
    #         if "metrics" in val_result:
    #             stats.update(val_result["metrics"])

    #     stats["epoch"] = epoch
    #     stats["split"] = split_name
        
    #     print("[AFTER_EVAL] Returning stats:", stats)
    #     return stats



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

        if is_dist_avail_and_initialized():
            dist.barrier()

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



    # @torch.no_grad()
    # def evaluation(self, model, data_loader, cuda_enabled=True):
    #     metric_logger = MetricLogger(delimiter="  ")
    #     header = "Evaluation"
    #     print_freq = 10

    #     total_loss = 0.0
    #     num_batches = 0
    #     all_outputs = []

    #     for samples in metric_logger.log_every(data_loader, print_freq, header):
    #         samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            
    #         eval_output = self.valid_step(model=model, samples=samples)  # <-- eval_output is a dict
            
    #         loss = eval_output.get("loss", None)
    #         output = eval_output.get("output", None)

    #         if loss is not None:
    #             total_loss += loss
            
    #         all_outputs.append(output)
    #         num_batches += 1

    #     if is_dist_avail_and_initialized():
    #         dist.barrier()

    #     avg_loss = total_loss / max(num_batches, 1)

    #     return {
    #         "loss": avg_loss,
    #         "output": all_outputs,
    #     }


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




    # # one way to compute retrieval metrics
    # def compute_retrieval_metrics_from_outputs(self, output_list, top_k=(1, 5, 10)):
    #     rgb_feats_list = []
    #     lidar_feats_list = []

    #     for out in output_list:
    #         inner_output = out["output"]  # Unpack the inner dict
    #         rgb_feats_list.append(inner_output["rgb_feats"])
    #         lidar_feats_list.append(inner_output["lidar_feats"]) 

    #     rgb_feats = torch.cat(rgb_feats_list, dim=0)      # [B_total, N, D]
    #     lidar_feats = torch.cat(lidar_feats_list, dim=0)  # [B_total, N, D]

    #     rgb_flat = rgb_feats.mean(dim=1)      # [B_total, D]
    #     lidar_flat = lidar_feats.mean(dim=1)  # [B_total, D]

    #     sim_matrix = torch.matmul(rgb_flat, lidar_flat.T)  # [B_total, B_total]

    #     # Now compute Recall@K and MRR
    #     B = sim_matrix.size(0)
    #     metrics = {}

    #     # Recall@K
    #     for k in top_k:
    #         hits = 0
    #         for i in range(B):
    #             topk = torch.topk(sim_matrix[i], k=k).indices
    #             if i in topk:
    #                 hits += 1
    #         metrics[f"recall@{k}"] = hits / B

    #     # MRR
    #     ranks = []
    #     for i in range(B):
    #         # descending rank
    #         sorted_indices = torch.argsort(sim_matrix[i], descending=True)
    #         rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
    #         ranks.append(1.0 / rank)
    #     metrics["mrr"] = sum(ranks) / B

    #     return metrics




    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        stats = self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=len(data_loader),
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )
        # Add epoch number to log
        # stats["epoch"] = epoch
        stats = {f"{k}": v for k, v in stats.items()}
        stats["epoch"] = epoch
        return stats
    


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


# ----------------------
# Simple Task Setup
# ----------------------

def setup_task(cfg):
    assert "task" in cfg.run_cfg, "Task name must be provided."
    task_name = cfg.run_cfg.task
    # task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    task = BaseTask()
    assert task is not None, f"Task {task_name} not properly registered."
    return task






