import torchvision as tv
import torch

from .customtransforms import PadMaxResize, DeployTransform


def get_transforms(min_img_size, max_img_size, mode="train", ):
    if mode == "train":
        transforms = tv.transforms.Compose([
            PadMaxResize(min_img_size, max_img_size,),
            tv.transforms.ConvertImageDtype(torch.float),
            tv.transforms.Normalize((0.7931, ), (0.1738, )),
        ])
    elif mode == "val":
        transforms = tv.transforms.Compose([
            PadMaxResize(min_img_size, max_img_size,),
            tv.transforms.ConvertImageDtype(torch.float),
            tv.transforms.Normalize((0.7931, ), (0.1738, )),
        ])
    elif mode == "deploy":
        val_transforms = tv.transforms.Compose([
            PadMaxResize(min_img_size, max_img_size,),
            tv.transforms.ConvertImageDtype(torch.float),
            tv.transforms.Normalize((0.7931, ), (0.1738, )),
        ])
        transforms = DeployTransform(
            val_transforms, max_img_size
        )
    else:
        raise NotImplementedError(f"mode {mode} not implemented")
    return transforms
