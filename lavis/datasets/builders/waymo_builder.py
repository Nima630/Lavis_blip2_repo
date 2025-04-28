# lavis/datasets/builders/waymo_builder.py

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.waymo_dataset import WaymoDataset
from lavis.common.registry import registry

@registry.register_builder("waymo")
class WaymoBuilder(BaseDatasetBuilder):
    train_dataset_cls = WaymoDataset
    eval_dataset_cls = WaymoDataset  # You can change this later if needed

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/waymo/defaults.yaml",
    }
