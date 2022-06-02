from typing import Tuple, Union
import torchvision as tv
import torch
import torchvision.transforms.functional as VF
from PIL import Image


class DeployTransform(object):
    def __init__(self, transform, max_img_size):
        self.transform = transform
        self.max_img_size = max_img_size

    def pad_image(self, img):
        """
            TODO: make pad image more clever, ie. don't pad all image to the max shape
        """
        img = self.transform(img)
        img = VF.pad(
            img,
            [0, 0, self.max_img_size[-1]-img.shape[-1], self.max_img_size[-2]-img.shape[-2]],
            0,
        )
        return img

    def __call__(self, img: Union[Image.Image, torch.Tensor]):
        if isinstance(img, Image.Image):
            img = VF.to_tensor(img)
        img = self.pad_image(img)
        return img


class PadMaxResize(tv.transforms.Resize):
    __max_iter__ = 10
    image_resizer = None
    def __init__(self, min_size: Tuple[int, int],  max_size: Tuple[int, int],
                 interpolation=tv.transforms.InterpolationMode.BILINEAR,
                ):
        super().__init__(1, interpolation,)
        self.min_size = min_size
        self.max_size = max_size
        # self.downsample = F.conv2d

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

    def forward(self, img: torch.Tensor):
        if img.shape[0] != 1:
            img = VF.to_tensor(VF.to_grayscale(img))
        # img = img.to(torch.uint8) - 255  # black background image
        # plt.imsave("black.png", img.numpy()[0], cmap="gray")
        # img = self.downsample(
        #     img.to(torch.float).unsqueeze(0),
        #     torch.ones(1, 1, 2, 2),  # in_chans, out_chans, kernel_size, kernel_size
        #     stride=3).squeeze(0)
        it = 0
        while not self.is_img_valid(img) and it < self.__max_iter__:
            img = self.pad_resize(img)
            it += 1
        assert it < self.__max_iter__, f"pad_resize match the maximum iter, img size: {img.shape}"
        return img


if __name__ == "__main__":
    ...
