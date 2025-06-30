
import os
import glob
import torch
from torch.utils.data import ConcatDataset
from PIL import Image
from collections import defaultdict
from  lavis.models.cam_to_bev import generate_bev_feature_map  # adjust to your actual import path
import argparse
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))



def get_grouped_multi_cam_files(camera_root, lidar_root):
    grouped = defaultdict(dict)  # sample_idx → {cam_name: path}

    for cam_name in os.listdir(camera_root):
        cam_dir = os.path.join(camera_root, cam_name)
        if not os.path.isdir(cam_dir) or not cam_name.startswith("CAM_"):
            continue

        for pt_file in glob.glob(os.path.join(cam_dir, "*.pt")):
            try:
                data = torch.load(pt_file, map_location='cpu')
                sample_idx = data["sample_idx"]
                grouped[sample_idx][cam_name] = pt_file
            except Exception as e:
                print(f"[WARN] Couldn't load {pt_file}: {e}")

    lidar_files = {
        os.path.splitext(os.path.basename(f))[0]: f
        for f in glob.glob(os.path.join(lidar_root, "*.pt"))
    }

    matched = []
    for sid, cams in grouped.items():
        if len(cams) == 6 and sid in lidar_files:
            matched.append((cams, lidar_files[sid]))

    print(f"[INFO] Found {len(matched)} matched samples with all 6 cameras + lidar.")
    return matched

class TorchMultiCameraDataset:
    def __init__(self, matched_samples):
        self.matched_samples = matched_samples

    def __len__(self):
        return len(self.matched_samples)

    # def __getitem__(self, idx):
    #     cam_dict, lidar_path = self.matched_samples[idx]

    #     try:
    #         cam_feats = []
    #         lidar2img_list = []
    #         img_shape_list = []
    #         cam_names = []

    #         # Ensure consistent order
    #         cam_order = sorted(cam_dict.keys())

    #         for cam_name in cam_order:
    #             data = torch.load(cam_dict[cam_name])
    #             cam_feats.append(data["feat"])
    #             lidar2img_list.append(data["lidar2img"])
    #             img_shape_list.append(data["img_shape"])
    #             cam_names.append(data["cam_name"])

    #         images = torch.cat(cam_feats, dim=0)  # [6, 1, 2048, 29, 50]
    #         lidar2img = torch.stack(lidar2img_list)  # [6, 4, 4]

    #         lidar = torch.load(lidar_path)["feat"]
    #         sample_idx = os.path.splitext(os.path.basename(lidar_path))[0]

    #     except Exception as e:
    #         print(f"[ERROR] Failed at idx={idx}: {e}")
    #         return None

    #     return {
    #         "images": images,
    #         "lidar": lidar,
    #         "lidar2img": lidar2img,
    #         "cam_names": cam_names,
    #         "img_shapes": img_shape_list,
    #         "sample_idx": sample_idx,
    #         "label": 1
    #     }

    #     # with bev projection
    
    
    # def __getitem__(self, idx):
    #     cam_dict, lidar_path = self.matched_samples[idx]

    #     try:
    #         cam_feats = []
    #         lidar2img_list = []
    #         img_shape_list = []
    #         cam_names = []

    #         cam_order = sorted(cam_dict.keys())

    #         for cam_name in cam_order:
    #             data = torch.load(cam_dict[cam_name])
    #             cam_feats.append(data["feat"])                    # [1, C, H, W]
    #             lidar2img_list.append(data["lidar2img"])          # [4, 4]
    #             img_shape_list.append(data["img_shape"])          # (H, W, 3)
    #             cam_names.append(data["cam_name"])

    #         images = torch.cat(cam_feats, dim=0)                  # [6, 1, C, H, W]
    #         lidar2img = torch.stack(lidar2img_list)               # [6, 4, 4]
    #         img_size = (images.shape[-2], images.shape[-1])       # Use image feature shape, e.g. (29, 50)

    #         # BEV Projection
    #         bev_feat = generate_bev_feature_map(images, lidar2img, img_size)  # [90, 90, C]

    #         lidar = torch.load(lidar_path)["feat"]  # [1, C_lidar, 90, 90]
    #         sample_idx = os.path.splitext(os.path.basename(lidar_path))[0]

    #     except Exception as e:
    #         print(f"[ERROR] Failed at idx={idx}: {e}")
    #         return None

    #     return {
    #         "bev_image": bev_feat,        # [90, 90, C]
    #         "lidar": lidar,               # [1, C_lidar, 90, 90]
    #         "sample_idx": sample_idx,
    #         "label": 1,
    #         "cam_names": cam_names
    #     }
    def __getitem__(self, idx):
        cam_dict, lidar_path = self.matched_samples[idx]

        try:
            cam_feats = []
            lidar2img_list = []
            img_shape_list = []
            cam_names = []

            cam_order = sorted(cam_dict.keys())

            for cam_name in cam_order:
                data = torch.load(cam_dict[cam_name])
                # print(f"[DEBUG_1] Keys in {cam_dict[cam_name]}: {list(data.keys())}")


                # Ensure feature is [1, 1, C, H, W]
                feat = data["feat"]
                if feat.dim() == 3:
                    feat = feat.unsqueeze(0).unsqueeze(0)
                elif feat.dim() == 4:
                    feat = feat.unsqueeze(0)
                cam_feats.append(feat)
                # cam_feats.append(data["feat"])                    # [1, C, H, W]
                # print(f"[DEBUG_2] cam_name={cam_name}, # feat.shape={cam_feats[0].shape}") # ([1, 2048, 29, 50])
                lidar2img_list.append(data["lidar2img"])          # [4, 4]
                img_shape_list.append(data["img_shape"])          # (H, W, 3)
                cam_names.append(data["cam_name"])

            images = torch.cat(cam_feats, dim=0)                  # [6, 1, C, H, W]
            print(f"[DEBUG_3] images.shape={images.shape}") # ([6, 2048, 29, 50])

            lidar2img = torch.stack(lidar2img_list)               # [6, 4, 4]
            print(f"[DEBUG_4] lidar2img.shape={lidar2img.shape}") # [6, 4, 4]

            img_size = (images.shape[-2], images.shape[-1])       # Use image feature shape, e.g. (29, 50)

            # BEV Projection
            bev_feat = generate_bev_feature_map(images, lidar2img, img_size)  # [90, 90, C]
            bev_feat = bev_feat.permute(2, 0, 1) 
            print(f"[DEBUG_5] bev_feat.shape={bev_feat.shape}") 

            lidar = torch.load(lidar_path)["feat"]  # [1, C_lidar, 90, 90]
            sample_idx = os.path.splitext(os.path.basename(lidar_path))[0]

        except Exception as e:
            print(f"[ERROR] Failed at idx={idx}: {e}")
            return None

        return {
            "bev_image": bev_feat,        # [90, 90, C]
            "lidar": lidar,               # [1, C_lidar, 90, 90]
            "sample_idx": sample_idx,
            "label": 1,
            "cam_names": cam_names
        }




    def collater(self, samples):
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return {}

        out = {}
        for key in samples[0]:
            vals = [s[key] for s in samples]
            if isinstance(vals[0], torch.Tensor):
                out[key] = torch.stack(vals)
            else:
                out[key] = vals
        return out

class TorchLidarDataset:
    def __init__(self, lidar_files):
        self.lidar_files = lidar_files

    def __getitem__(self, idx):
        path = self.lidar_files[idx]
        try:
            data = torch.load(path)
            lidar = data['feat']  # Updated to match your file format
        except Exception as e:
            print(f"[LiDAR] Error loading {path}: {e}")
            return None

        return {
            "lidar": lidar,
            "image_id": os.path.splitext(os.path.basename(path))[0]
        }

    def __len__(self):
        return len(self.lidar_files)

# creates train/val/test splits from config
class NuScenesDatasetBuilder:
    def __init__(self, cfg):
        self.config = cfg
    def build_datasets(self):
        build_info = self.config.datasets_cfg["nuscenes"].build_info
        datasets = {}

        for split in ["train", "val", "test"]:
            if split in self.config.run_cfg.train_splits + \
                    self.config.run_cfg.valid_splits + \
                    self.config.run_cfg.test_splits:

                camera_root = build_info["camera"][split].storage
                lidar_root = build_info["lidar"][split].storage

                matched_samples = get_grouped_multi_cam_files(camera_root, lidar_root)

                # Each sample now includes all 6 cameras + lidar + lidar2img
                dataset = TorchMultiCameraDataset(matched_samples)
                datasets[split] = dataset

        return datasets


def get_matched_files(camera_root, lidar_root):
    camera_files = {
        os.path.splitext(os.path.basename(f))[0]: f
        for f in glob.glob(os.path.join(camera_root, "*.pt"))
    }
    lidar_files = {
        os.path.splitext(os.path.basename(f))[0]: f
        for f in glob.glob(os.path.join(lidar_root, "*.pt"))
    }

    matched_keys = sorted(set(camera_files) & set(lidar_files))
    matched_camera_files = [camera_files[k] for k in matched_keys]
    matched_lidar_files = [lidar_files[k] for k in matched_keys]

    return matched_camera_files, matched_lidar_files

