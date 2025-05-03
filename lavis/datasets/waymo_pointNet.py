import logging
import os
import glob
import pandas as pd
import torch
import numpy as np
import ast
from PIL import Image
from io import BytesIO
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
import warnings
import torch.distributed as dist
from lavis.common.dist_utils import is_dist_avail_and_initialized, is_main_process
from lavis.common import utils
from lavis.processors.processor import BaseProcessor, Blip2ImageTrainProcessor
from omegaconf import OmegaConf, ListConfig
import matplotlib.pyplot as plt


class WaymoCameraDataset:
    def __init__(self, vis_processor, camera_files):
        self.vis_processor = vis_processor
        self.camera_files = camera_files
        
    def __getitem__(self, idx):
        parquet_file = self.camera_files[idx]
        try:
            df = pd.read_parquet(parquet_file)
            image_bytes = df['[CameraImageComponent].image'].iloc[0]
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image = self.vis_processor(image)
        except Exception as e:
            print(f"[CameraImage] Error loading image from file {parquet_file}: {e}")
            return None

        return {
            "image": image,
            "image_id": os.path.basename(parquet_file)
        }

    def __len__(self):
        return len(self.camera_files)

    def collater(self, samples):
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return {}
        batch = {
            k: torch.stack([s[k] for s in samples]) if isinstance(samples[0][k], torch.Tensor) else [s[k] for s in samples]
            for k in samples[0]
        }
        return batch



class WaymoLidarDataset:
    def __init__(self, vis_processor, lidar_files):
        self.vis_processor = vis_processor
        self.lidar_files = lidar_files

    def __getitem__(self, idx):
        parquet_file = self.lidar_files[idx]
        try:
            df = pd.read_parquet(parquet_file)
            row = df.iloc[0]

            raw_vals = row['[LiDARComponent].range_image_return1.values']
            raw_shape = row['[LiDARComponent].range_image_return1.shape']

            values = np.array(ast.literal_eval(raw_vals) if isinstance(raw_vals, str) else raw_vals)
            shape = ast.literal_eval(raw_shape) if isinstance(raw_shape, str) else raw_shape
            data = values.reshape(shape)  # (H, W, 4)

            range_ = data[:, :, 0]
            intensity = data[:, :, 1]
            elongation = data[:, :, 2]
            mask = data[:, :, 0] > 0  # Keep all valid points based on non-zero range 

            # Convert spherical to 3D points
            h, w = range_.shape
            inclinations = np.linspace(np.radians(2.4), np.radians(-17.6), h)  # Approx Waymo vertical FoV
            azimuths = np.linspace(-np.pi, np.pi, w)
            azimuth_grid, incl_grid = np.meshgrid(azimuths, inclinations)

            r = range_[mask]
            x = r * np.cos(incl_grid[mask]) * np.cos(azimuth_grid[mask])
            y = r * np.cos(incl_grid[mask]) * np.sin(azimuth_grid[mask])
            z = r * np.sin(incl_grid[mask])

            extra_feats = np.stack([intensity[mask], elongation[mask]], axis=-1)
            points = np.stack([x, y, z], axis=-1)
            point_feats = np.concatenate([points, extra_feats], axis=-1)  # [N, 5]

            point_tensor = torch.tensor(point_feats, dtype=torch.float32)

        except Exception as e:
            print(f"[LiDAR] Failed to load {parquet_file}: {e}")
            return None

        return {
            "lidar": point_tensor,  # [N, 5] = xyz + intensity + elongation
            "image_id": os.path.basename(parquet_file)
        }

    def __len__(self):
        return len(self.lidar_files)

    def collater(self, samples):
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return {}
        batch = {
            k: torch.stack([s[k] for s in samples]) if isinstance(samples[0][k], torch.Tensor) else [s[k] for s in samples]
            for k in samples[0]
        }
        return batch




class WrappedConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        assert len(datasets) == 2, "Expecting camera and lidar datasets"
        super().__init__(datasets)
        self.camera_dataset = datasets[0]
        self.lidar_dataset = datasets[1]

    def __getitem__(self, idx):
        # print(f"[DEBUG] WrappedConcatDataset __getitem__({idx})")
        try:
            if idx >= len(self.camera_dataset) or idx >= len(self.lidar_dataset):
                return None
            cam_sample = self.camera_dataset[idx]
            lidar_sample = self.lidar_dataset[idx]
            # print(f"[DEBUG] cam_sample keys: {cam_sample.keys() if cam_sample else 'None'}")
            # print(f"[DEBUG] lidar_sample keys: {lidar_sample.keys() if lidar_sample else 'None'}")
            if cam_sample is None or lidar_sample is None:
                return None
            return {**cam_sample, **lidar_sample, "label": 1}  # dummy label
        except Exception as e:
            print(f"[ERROR] Dataset error at index {idx}: {e}")
            return None

    def __len__(self):
        return min(len(self.camera_dataset), len(self.lidar_dataset))

    def collater(self, samples):
        # print(f"[DEBUG] collater received {len(samples)} samples")
        
        for i, s in enumerate(samples):
            print(f"  sample {i}: keys = {s.keys() if s else 'None'}")
        
        samples = [s for s in samples if s is not None]

        if len(samples) == 0:
            print("[WARNING] All samples were None!")
            return {}
        batch = self.camera_dataset.collater(samples)
        batch.update({"lidar": torch.stack([s["lidar"] for s in samples])})
        batch["label"] = torch.tensor([s["label"] for s in samples])
        # print(f"[DEBUG] Final collated batch keys+++++++++++++++++++: {batch.keys()}")
        return batch


class WaymoDatasetBuilder:
    def __init__(self, cfg):
        self.config = cfg
        # print("---------------------------- self.config",self.config)
        # cfg.datasets_cfg["waymo"]
        self.data_type = list(cfg.datasets_cfg["waymo"].data_type)

    # def build_datasets(self):
    #     build_info = self.config.build_info
    #     vis_processor = Blip2ImageTrainProcessor.from_config(self.config.vis_processor.train)

    #     # camera_dataset = WaymoCameraDataset(vis_processor, build_info["camera"].storage)
    #     # lidar_dataset = WaymoLidarDataset(vis_processor, build_info["lidar"].storage)
    #     camera_root = build_info["camera"].storage
    #     lidar_root = build_info["lidar"].storage

    #     matched_camera_files, matched_lidar_files = self.get_matched_files(camera_root, lidar_root)

    #     camera_dataset = WaymoCameraDataset(vis_processor, matched_camera_files)
    #     lidar_dataset = WaymoLidarDataset(vis_processor, matched_lidar_files)


    #     return {
    #         "train": WrappedConcatDataset([camera_dataset, lidar_dataset])
    # }
    
    
    # def build_datasets(self):
    #     build_info = self.config.build_info
    #     vis_processor = Blip2ImageTrainProcessor.from_config(self.config.vis_processor.train)

    #     datasets = {}

    #     for split in ["train", "val", "test"]:
    #         if split in self.config.run.train_splits + self.config.run.valid_splits + self.config.run.test_splits:
    #             camera_root = build_info["camera"][split].storage
    #             lidar_root = build_info["lidar"][split].storage

    #             matched_camera_files, matched_lidar_files = self.get_matched_files(camera_root, lidar_root)

    #             camera_dataset = WaymoCameraDataset(vis_processor, matched_camera_files)
    #             lidar_dataset = WaymoLidarDataset(vis_processor, matched_lidar_files)

    #             datasets[split] = WrappedConcatDataset([camera_dataset, lidar_dataset])

    #     return datasets


    def build_datasets(self):
        build_info = self.config.datasets_cfg["waymo"].build_info
        # print("[DEBUG] build_info ", build_info)

        # print("[DEBUG] build_info keys:", build_info.keys())
        vis_processor = Blip2ImageTrainProcessor.from_config(self.config.datasets_cfg["waymo"].vis_processor.train)

        datasets = {}

        # Let's print what's inside self.config.run
        # print("[DEBUG] config.run keys:", self.config.run_cfg.keys())

        for split in ["train", "val", "test"]:
            # Print current split
            # print(f"[DEBUG] Processing split: {split}")

            # Print which splits we have
            # print(f"[DEBUG] train_splits: {self.config.run_cfg.train_splits}")
            # print(f"[DEBUG] valid_splits: {self.config.run_cfg.valid_splits}")
            # print(f"[DEBUG] test_splits: {self.config.run_cfg.test_splits}")

            if split in self.config.run_cfg.train_splits + self.config.run_cfg.valid_splits + self.config.run_cfg.test_splits:
                # print(f"[DEBUG] Split {split} is in config. Proceeding...")

                # Debugging build_info structure
                # print(f"[DEBUG] build_info['camera'] keys: {build_info['camera'].keys()}")
                # print(f"[DEBUG] build_info['lidar'] keys: {build_info['lidar'].keys()}")

                camera_root = build_info["camera"][split].storage
                lidar_root = build_info["lidar"][split].storage

                # print(f"[DEBUG] Camera root for {split}: {camera_root}")
                # print(f"[DEBUG] LiDAR root for {split}: {lidar_root}")

                matched_camera_files, matched_lidar_files = self.get_matched_files(camera_root, lidar_root)

                camera_dataset = WaymoCameraDataset(vis_processor, matched_camera_files)
                lidar_dataset = WaymoLidarDataset(vis_processor, matched_lidar_files)

                datasets[split] = WrappedConcatDataset([camera_dataset, lidar_dataset])

        return datasets


    def get_matched_files(self, camera_root, lidar_root):
        camera_files = {
            os.path.splitext(os.path.basename(f))[0]: f
            for f in glob.glob(os.path.join(camera_root, "*.parquet"))
        }
        lidar_files = {
            os.path.splitext(os.path.basename(f))[0]: f
            for f in glob.glob(os.path.join(lidar_root, "*.parquet"))
        }

        # Get intersection of file keys
        matched_keys = sorted(set(camera_files) & set(lidar_files))
        # matched_keys = sorted(set(camera_files) & set(lidar_files))[:300]
        matched_camera_files = [camera_files[k] for k in matched_keys]
        matched_lidar_files = [lidar_files[k] for k in matched_keys]

        return matched_camera_files, matched_lidar_files


if __name__ == "__main__":
    from glob import glob

    # Paths
    lidar_data_path = "lavis/my_datasets/waymo/train/lidar"
    lidar_files = sorted(glob(os.path.join(lidar_data_path, "*.parquet")))

    # Dummy processor to keep structure consistent
    dummy_processor = lambda x: x

    # Instantiate dataset
    lidar_dataset = WaymoLidarDataset(
        vis_processor=dummy_processor,
        lidar_files=lidar_files
    )
    print(f"[INFO] Loaded {len(lidar_dataset)} LiDAR samples.")

    valid_count = sum(1 for i in range(len(lidar_dataset)) if lidar_dataset[i] and lidar_dataset[i]["lidar"].shape[0] > 0)
    print(f"\n[INFO] Number of valid LiDAR samples (with points): {valid_count} / {len(lidar_dataset)}")


    

    # for i in range(min(5, len(lidar_dataset))):
    #     sample = lidar_dataset[i]
    #     if sample:
    #         print(f"[{i}] Image ID: {sample['image_id']}")
    #         print(f"     Point Cloud Shape: {sample['lidar'].shape}")  # (N, 5)
    #         print(f"     First 3 points:\n{sample['lidar'][:3]}")














