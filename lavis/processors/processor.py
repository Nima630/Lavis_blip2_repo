import re
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class BaseProcessor:
    def __init__(self):
        self.transform = lambda x: x

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        return cls()

    def build(self, **kwargs):
        cfg = OmegaConf.create(kwargs)
        return self.from_config(cfg)


class BlipCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        return self.prompt + self.pre_caption(caption)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()
        return cls(
            prompt=cfg.get("prompt", ""),
            max_words=cfg.get("max_words", 50)
        )

    def pre_caption(self, caption):
        caption = re.sub(r"([.!\"()*#:;~])", " ", caption.lower())
        caption = re.sub(r"\s{2,}", " ", caption)
        caption = caption.rstrip("\n").strip(" ")
        words = caption.split(" ")
        if len(words) > self.max_words:
            caption = " ".join(words[: self.max_words])
        return caption


class Blip2ImageTrainProcessor(BaseProcessor):
    def __init__(self, image_size=364, mean=None, std=None, min_scale=0.5, max_scale=1.0):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(min_scale, max_scale),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize
        ])

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()
        return cls(
            image_size=cfg.get("image_size", 364),
            mean=cfg.get("mean", None),
            std=cfg.get("std", None),
            min_scale=cfg.get("min_scale", 0.5),
            max_scale=cfg.get("max_scale", 1.0)
        )
