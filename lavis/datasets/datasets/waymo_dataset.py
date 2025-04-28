# lavis/datasets/datasets/waymo_dataset.py

import os
from PIL import Image
import pandas as pd
from lavis.datasets.datasets.base_dataset import BaseDataset
# from lavis.datasets.datasets.waymo_utils import load_camera_image, load_lidar_range_image  # assumes you've written these

class WaymoDataset(BaseDataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        # Load LiDAR .parquet once (or implement streaming if it's large)
        lidar_path = os.path.join(vis_root, "lidar.parquet")
        self.lidar_df = pd.read_parquet(lidar_path)

    def __getitem__(self, index):
        ann = self.annotation[index]

        # Load RGB image
        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        # Load LiDAR range image (your util handles reshaping/parsing)
        lidar_tensor = load_lidar_range_image(self.lidar_df, index)

        return {
            "image": image,
            "lidar": lidar_tensor,
            "instance_id": ann.get("instance_id", str(index))
        }


















# import os
# import pandas as pd
# import numpy as np
# from PIL import Image
# from io import BytesIO
# import torch
# import ast
# from torch.utils.data import Dataset

# class WaymoDataset(Dataset):
#     def __init__(self, camera_parquet_path, lidar_parquet_path, vis_processor=None, text_processor=None):
#         """
#         Args:
#             camera_parquet_path: path to camera_image.parquet
#             lidar_parquet_path: path to lidar.parquet
#             vis_processor: optional image transform (e.g. resize, normalize)
#             text_processor: optional caption/text processor (for compatibility)
#         """
#         self.camera_df = pd.read_parquet(camera_parquet_path)
#         self.lidar_df = pd.read_parquet(lidar_parquet_path)
#         self.vis_processor = vis_processor
#         self.text_processor = text_processor  # can be None if not needed

#         assert len(self.camera_df) == len(self.lidar_df), "Camera and LiDAR data must be aligned"

#     def __len__(self):
#         return len(self.camera_df)

#     def __getitem__(self, index):
#         # ==== Load Camera Image ====
#         image_bytes = self.camera_df['[CameraImageComponent].image'].iloc[index]
#         image = Image.open(BytesIO(image_bytes)).convert("RGB")
#         if self.vis_processor:
#             image = self.vis_processor(image)

#         # ==== Load LiDAR Range Image ====
#         raw_values = self.lidar_df['[LiDARComponent].range_image_return1.values'].iloc[index]
#         raw_shape = self.lidar_df['[LiDARComponent].range_image_return1.shape'].iloc[index]

#         # Convert values and shape if needed
#         if isinstance(raw_values, str):
#             values = np.array(ast.literal_eval(raw_values))
#         else:
#             values = np.array(raw_values)

#         if isinstance(raw_shape, str):
#             shape = ast.literal_eval(raw_shape)
#         else:
#             shape = raw_shape

#         range_image = values.reshape(shape)
#         lidar_tensor = torch.tensor(range_image).float()  # shape: (H, W, C)

#         # Reorder to (C, H, W) if needed
#         if lidar_tensor.ndim == 3:
#             lidar_tensor = lidar_tensor.permute(2, 0, 1)

#         return {
#             "image": image,                 # Processed RGB image
#             "lidar": lidar_tensor,          # Raw or processed LiDAR tensor
#             "instance_id": str(index),      # Optional unique ID
#         }
