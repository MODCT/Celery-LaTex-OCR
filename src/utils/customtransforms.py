from typing import Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
import torchvision as tv
import torch
import random
import torch.nn as nn
import torchvision.transforms.functional as VF
from PIL import Image


class DeployTransform(object):
    _buckets_ = [32, 64, 96, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896]
    def __init__(self, transform, max_img_size):
        self.transform = transform
        # self.max_img_size = max_img_size
        self.buckets = [[i, j] for i in self._buckets_ for j in self._buckets_]

    def pad_image(self, img: torch.Tensor):
        """
            TODO: make pad image more clever, ie. don't pad all image to the max shape
        """
        # img: C H W
        # new_size = self.get_new_size(img.shape[1:])
        new_size = (192, 896)
        pw = new_size[-1] - img.shape[-1]
        ph = new_size[-2] - img.shape[-2]
        # img = np.pad(img, ((0, 0), (0, ph), (0, pw)), "constant", constant_values=(0, ))
        img = VF.pad(img, [0, 0, pw, ph], 0,)
        return img

    def get_new_size(self, old_size: Tuple[int]):
        d1, d2 = old_size
        for (d1_b, d2_b) in self.buckets:
            if d1_b >= d1 and d2_b >= d2:
                return d1_b, d2_b
        # no match, return max (last one)
        return self.buckets[-1]

    # def pad_image(self, img):
    #     """
    #         TODO: make pad image more clever, ie. don't pad all image to the max shape
    #     """
    #     img = self.transform(img)
    #     img = VF.pad(
    #         img,
    #         [0, 0, self.max_img_size[-1]-img.shape[-1], self.max_img_size[-2]-img.shape[-2]],
    #         0,
    #     )
    #     return img

    def __call__(self, img: Union[Image.Image, torch.Tensor]):
        if isinstance(img, Image.Image):
            img = torch.Tensor(np.asarray(img))
        img = self.transform(img)
        img = self.pad_image(img)
        return img


class PadMaxResize(nn.Module):
    __max_iter__ = 10
    image_resizer = None
    def __init__(self, min_size: Tuple[int, int],  max_size: Tuple[int, int],
                 interpolation=tv.transforms.InterpolationMode.BILINEAR,
                ):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.interpolation = interpolation

    def pad_resize(self, img: torch.Tensor):
        _, h, w = img.shape
        mxh, mxw = self.max_size
        mnh, mnw = self.min_size
        # height
        if h > mxh:
            ratio_h = h / mxh
        elif h < mnh:
            ratio_h = -1
        else:
            ratio_h = 1
        # width
        if w > mxw:
            ratio_w = w / mxw
        elif w < mnw:
            ratio_w =  -1
        else:
            ratio_w = 1
        if ratio_h == 1 and ratio_w == 1:
            return img
        # pad first
        if ratio_h == -1 or ratio_w == -1:
            pw = mnw - w if ratio_w == -1 else 0
            ph = mnh - h if ratio_h == -1 else 0
            img = VF.pad(img, [0, 0, pw, ph], fill=255)
        c, h, w = img.shape
        if ratio_h > 1 or ratio_w > 1:
            ratio = max(ratio_h, ratio_w)
            size = (int(h/ratio), int(w/ratio))
            img = VF.resize(img, size, self.interpolation)
        return img

    def is_img_valid(self, img: torch.Tensor):
        _, h, w = img.shape
        c = False
        if self.min_size[0] <= h <= self.max_size[0] and self.min_size[1] <= w <= self.max_size[1]:
            c = True
        return c

    def random_resize(self, img: torch.Tensor):
        # ratio in [0.5, 2]
        ratio = random.randint(5, 20) / 10
        # interp = tv.transforms.InterpolationMode.BILINEAR if ratio < 1 else tv.transforms.InterpolationMode.LANCZOS
        newsize = (int(img.shape[-2]*ratio), int(img.shape[-1]*ratio))
        img = VF.resize(img, newsize, self.interpolation)
        return img

    def crop_bbox(self, img: torch.Tensor):
        if len(img.shape) == 3:
            img = img[0]
        bg = torch.full_like(img, 255)
        diff = (img - bg).nonzero(as_tuple=True)
        top, left = diff[0].min(), diff[1].min()
        bot, right = diff[0].max(), diff[1].max()
        img = VF.crop(img, top, left, bot-top+1, right-left+1)
        return img.unsqueeze(0)

    def save_img(self, img: torch.Tensor, name="test.png"):
        plt.imsave(name, img.numpy()[0], cmap="gray")

    def forward(self, img: torch.Tensor):
        if img.shape[0] != 1:
            img: np.ndarray = img.numpy()[0]
            img = np.asarray(Image.fromarray(img.astype(np.uint8), mode="L"), dtype=np.float32)
            img = torch.from_numpy(img).unsqueeze(0)
        # img = img.to(torch.uint8) - 255  # black background image
        # self.save_img(img, "original.png")
        img = self.crop_bbox(img)
        # self.save_img(img, "crop_box.png")
        img = self.random_resize(img)
        self.save_img(img, "random_resize.png")
        it = 0
        while not self.is_img_valid(img) and it < self.__max_iter__:
            img = self.pad_resize(img)
            it += 1
        assert it < self.__max_iter__, f"pad_resize match the maximum iter, img size: {img.shape}"
        return img


if __name__ == "__main__":
    ...