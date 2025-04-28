"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder, MultiModalDatasetBuilder
# from lavis.datasets.datasets.capfilt_dataset import CapFiltCaptionInstructDataset, CapFiltCaptionDataset
from lavis.datasets.datasets.coco_caption_datasets import (
    COCOCapDataset,
    COCOCapInstructDataset,
    COCOCapEvalDataset,
    NoCapsEvalDataset,
)

from lavis.common.registry import registry

@registry.register_builder("coco_caption")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset
    eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_cap.yaml",
    }

@registry.register_builder("coco_caption_instruct")
class COCOCapInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapInstructDataset
    eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_cap_instruct.yaml",
    }


# @registry.register_builder("flickr30k_caption")
class Flickr30kCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset
    eval_dataset_cls = COCOCapEvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/flickr30k/defaults_cap.yaml",
    }

# @registry.register_builder("flickr30k_caption_instruct")
class Flickr30kCapInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapInstructDataset
    eval_dataset_cls = COCOCapEvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/flickr30k/defaults_cap_instuct.yaml",
    }

# @registry.register_builder("nocaps")
class COCOCapBuilder(BaseDatasetBuilder):
    eval_dataset_cls = NoCapsEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nocaps/defaults.yaml",
    }
