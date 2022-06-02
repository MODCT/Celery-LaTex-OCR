import torchvision as tv
import torch

from .utils.customtransforms import PadMaxResize, DeployTransform
from .config import Config


CONF = Config()


train_transforms = tv.transforms.Compose([
    PadMaxResize(CONF.min_img_size, CONF.max_img_size,),
    tv.transforms.ConvertImageDtype(torch.float),
    tv.transforms.Normalize((0.7931, ), (0.1738, )),
])


val_transforms = tv.transforms.Compose([
    PadMaxResize(CONF.min_img_size, CONF.max_img_size,),
    tv.transforms.ConvertImageDtype(torch.float),
    tv.transforms.Normalize((0.7931, ), (0.1738, )),
])


deploy_transforms = DeployTransform(
    val_transforms, CONF.max_img_size
)
