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
        # print("[TRACE] train_step in base_task.py")
        # print("  sample keys:", samples.keys())
        # print("  image shape:", samples.get("image", None).shape if "image" in samples else "missing")
        # print("  lidar shape:", samples.get("lidar", None).shape if "lidar" in samples else "missing")
        output = model(samples)
        # print("[TRACE] train_step in base_task.py after forward pass")
        loss_dict = {k: v for k, v in output.items() if "loss" in k}
        return output["loss"], loss_dict

    # def valid_step(self, model, samples):
    #     raise NotImplementedError



    def valid_step(self, model, samples):
        model.eval()
        samples = prepare_sample(samples, cuda_enabled=True)

        use_amp = True  # or however you track AMP usage (usually from config)

        with torch.no_grad():
            with autocast("cuda", enabled=use_amp):
                output = model(samples)

        if "loss" in output:
            loss = output["loss"].item()
        else:
            loss = None

        print(f"[VALIDATION] Batch validation loss: {loss}")

        return {"loss": loss, "output": output}


    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        print_freq = 10
        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

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
            # print("  sample keys: ++++++++++++++++++++++++++++++++++++", samples)
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
    # task_name = cfg.run_cfg.task
    # task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    task = BaseTask()
    assert task is not None, f"Task {task_name} not properly registered."
    return task






