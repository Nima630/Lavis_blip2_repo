import logging
import os
import shutil
import warnings

import lavis.common.utils as utils
import torch.distributed as dist
from lavis.common.dist_utils import is_dist_avail_and_initialized, is_main_process
from lavis.processors.processor import BaseProcessor, Blip2ImageTrainProcessor, BlipCaptionProcessor
from omegaconf import OmegaConf
from torchvision.datasets.utils import download_url


class MultiModalDatasetBuilder:
    train_dataset_cls = None
    eval_dataset_cls = None

    def __init__(self, cfg=None):
        if cfg is None:
            self.config = load_dataset_config(self.default_config_path())
        elif isinstance(cfg, str):
            self.config = load_dataset_config(cfg)
        else:
            self.config = cfg

        self.data_type = self.config.data_type
        if isinstance(self.data_type, str):
            self.data_type = [self.data_type]

        self.text_processors = {}
        self.processors = {}

    def build_datasets(self):
        print("\n>>> [DEBUG] Dataset builder config:", self.config)
        if is_main_process():
            self._download_data()
        if is_dist_avail_and_initialized():
            dist.barrier()

        logging.info("Building datasets...")
        datasets = self.build()
        return datasets

    def build_processors(self):
        self.text_processors = self._build_processor("text_processor")
        self.processors = {
            split: {
                modality: self._build_proc_from_cfg(
                    self.config.get(f"{'vis' if 'image' in modality else modality}_processor", {}).get(split)
                ) for modality in self.data_type
            } for split in ["train", "eval"]
        }

    def _build_processor(self, cfg_name):
        cfg = self.config.get(cfg_name)
        return {
            split: self._build_proc_from_cfg(cfg.get(split)) if cfg else None
            for split in ["train", "eval"]
        }

    def _download_multimodal(self, modality):
        storage_path = utils.get_cache_path(self.config.build_info.get(modality).storage)
        if not os.path.exists(storage_path):
            warnings.warn(f"The specified path {storage_path} for {modality} inputs does not exist.")

    def _download_data(self):
        self._download_ann()
        for modality in self.data_type:
            self._download_multimodal(modality)

    @staticmethod
    def _build_proc_from_cfg(cfg):
        if cfg is None:
            return BaseProcessor()
        name = cfg.name
        if name == "blip2_image_train":
            return Blip2ImageTrainProcessor.from_config(cfg)
        if name == "blip_caption":
            return BlipCaptionProcessor.from_config(cfg)
        raise ValueError(f"Unknown processor: {name}")

    def _download_data(self):
        for modality in self.data_type:
            self._download_modality(modality)

    def _download_modality(self, modality):
        path = utils.get_cache_path(self.config.build_info.get(modality).storage)
        if not os.path.exists(path):
            warnings.warn(f"The specified path {path} for {modality} inputs does not exist.")

    def build(self):
        self.build_processors()
        build_info = self.config.build_info
        datasets = {}

        for split in ["train", "val", "test"]:
            is_train = split == "train"
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls

            dataset_args = {
                f"{modality}_root": utils.get_cache_path(build_info[modality].storage)
                for modality in self.data_type
            }

            dataset_args.update({
                f"{modality}_processor": self.processors[split][modality]
                for modality in self.data_type
            })

            dataset_args["text_processor"] = self.text_processors[split]
            dataset_args["ann_paths"] = ["dummy_ann.json"]  # Use dummy path for now
            dataset_args["modalities"] = self.data_type

            datasets[split] = dataset_cls(**dataset_args)

        return datasets

    @classmethod
    def default_config_path(cls, type="default"):
        return utils.get_abs_path(cls.DATASET_CONFIG_DICT[type])

def load_dataset_config(cfg_path):
    cfg = OmegaConf.load(cfg_path).datasets
    return next(iter(cfg.values()))
