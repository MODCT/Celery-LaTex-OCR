import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.utils import to_2tuple


class PositionalEmbedding(nn.Module):
    # from bert

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, emb_dim)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x):
        n = torch.arange(x.shape[1], device=x.device)
        return self.emb(n)[None, :, :]


class SegmentEmbedding(nn.Module):
    # method form bert
    def __init__(self, embed_dim, num_patches):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        # normal init
        nn.init.kaiming_normal_(self.pos_embed, nonlinearity="relu")

    def forward(self, x):
        x += self.pos_embed[:, : x.shape[1]]
        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding
    Modified from timm.models.layers.patch_embed
    URL: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/patch_embed.py
    LICENSE:  Apache-2.0 license
    Date: 2022.05.07
    Use Conv2d to make patches
    img_size is used to determin the `MAX` number of patches (num_patches)
    the output result of `forward` will adjust according to the input shape
    `NOTE`: the input image shape can't < patch_size
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        # patch_dim = in_chans * patch_size[0] * patch_size[1]

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            # nn.Linear(patch_dim*, embed_dim),
        )

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        # pad if H or W can't be divisable by patch_size
        pad_r, pad_b = 0, 0
        ph, pw = self.patch_size
        # print(x.shape, ph, pw)
        if H % ph != 0:
            pad_b = (int(H / ph) + 1) * ph - H
        if W % pw != 0:
            pad_r = (int(W / pw) + 1) * pw - W
        if any([pad_r != 0, pad_b != 0]):
            # x0 = VF.pad(x, [0, 0, pad_r, pad_b])
            # x1 = F.pad(x, [0, pad_r, 0, pad_b, 0, 0, 0, 0])
            # print(pad_r, pad_b, x0.shape, x1.shape)
            # print(torch.equal(x0, x1))
            x = F.pad(x, [0, pad_r, 0, pad_b, 0, 0, 0, 0])
        x = self.proj(x.float())
        # print(x.shape)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x
