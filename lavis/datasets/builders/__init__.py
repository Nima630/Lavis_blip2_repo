"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""





from lavis.datasets.builders.caption_builder import (
    COCOCapBuilder,
    Flickr30kCapBuilder,
    Flickr30kCapInstructBuilder,
)
from lavis.datasets.builders.waymo_builder import WaymoBuilder  # Add custom datasets

from lavis.datasets.builders.base_dataset_builder import load_dataset_config

# Manual builder mapping
builder_map = {
    "coco_caption": COCOCapBuilder,
    "flickr30k_caption": Flickr30kCapBuilder,
    "flickr30k_caption_instruct": Flickr30kCapInstructBuilder,
    "waymo": WaymoBuilder,
}


def load_dataset(name, cfg_path=None, vis_path=None, data_type=None):
    """
    Loads dataset from builder name.

    >>> dataset = load_dataset("coco_caption", cfg=None)
    >>> print(dataset.keys())  # ['train', 'val', 'test']
    """
    if cfg_path is None:
        cfg = None
    else:
        cfg = load_dataset_config(cfg_path)

    try:
        builder_cls = builder_map[name]
        builder = builder_cls(cfg)
    except KeyError:
        print(
            f"[ERROR] Dataset '{name}' not found.\nAvailable datasets: "
            + ", ".join(builder_map.keys())
        )
        exit(1)

    # Optional: Override visual path manually
    if vis_path is not None:
        if data_type is None:
            data_type = builder.config.data_type

        assert data_type in builder.config.build_info, (
            f"Invalid data_type '{data_type}' for dataset '{name}'."
        )
        builder.config.build_info[data_type].storage = vis_path

    return builder.build_datasets()


class DatasetZoo:
    def __init__(self):
        self.dataset_zoo = {
            k: list(v.DATASET_CONFIG_DICT.keys())
            for k, v in builder_map.items()
        }

    def get_names(self):
        return list(self.dataset_zoo.keys())


dataset_zoo = DatasetZoo()




# from lavis.datasets.builders.base_dataset_builder import load_dataset_config
# from lavis.datasets.builders.caption_builder import (
#     COCOCapBuilder,
#     Flickr30kCapBuilder,
#     Flickr30kCapInstructBuilder)


# from lavis.common.registry import registry

# __all__ = [
#     "COCOCapBuilder",
#     "Flickr30kCapBuilder",
#     "Flickr30kCapInstructBuilder"
# ]


# def load_dataset(name, cfg_path=None, vis_path=None, data_type=None):
#     """
#     Example

#     >>> dataset = load_dataset("coco_caption", cfg=None)
#     >>> splits = dataset.keys()
#     >>> print([len(dataset[split]) for split in splits])

#     """
#     if cfg_path is None:
#         cfg = None
#     else:
#         cfg = load_dataset_config(cfg_path)

#     try:
#         builder = registry.get_builder_class(name)(cfg)
#     except TypeError:
#         print(
#             f"Dataset {name} not found. Available datasets:\n"
#             + ", ".join([str(k) for k in dataset_zoo.get_names()])
#         )
#         exit(1)

#     if vis_path is not None:
#         if data_type is None:
#             # use default data type in the config
#             data_type = builder.config.data_type

#         assert (
#             data_type in builder.config.build_info
#         ), f"Invalid data_type {data_type} for {name}."

#         builder.config.build_info.get(data_type).storage = vis_path

#     dataset = builder.build_datasets()
#     return dataset


# class DatasetZoo:
#     def __init__(self) -> None:
#         self.dataset_zoo = {
#             k: list(v.DATASET_CONFIG_DICT.keys())
#             for k, v in sorted(registry.mapping["builder_name_mapping"].items())
#         }

#     def get_names(self):
#         return list(self.dataset_zoo.keys())


# dataset_zoo = DatasetZoo()










