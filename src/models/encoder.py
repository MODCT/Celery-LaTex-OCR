from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath

from ..utils.utils import to_2tuple
from .embed import PatchEmbed, PositionalEmbedding


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LatexVITEncoder(nn.Module):
    __model_name__ = "vit"

    def __init__(self, img_size: Union[Tuple[int, int], int]=224, patch_size: Union[Tuple[int, int], int]=16,
                 in_chans: int=1, embed_dim: int=768, num_heads: int=8, drop_rate: float=0.,
                 depth: int=4, convdepth: int=2, norm_layer=None, act_layer=None, ff_dim=2048,
                 kernel_size: int=7,
                ):
        super().__init__()
        self.im_height, self.im_width = to_2tuple(img_size)
        self.patch_height, self.patch_width = to_2tuple(patch_size)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, flatten=False, norm_layer=nn.LayerNorm,)
        num_patches = (self.im_height // self.patch_height) * (self.im_width // self.patch_width)
        # num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_embedding = PositionalEmbedding(embed_dim, num_patches+1)
        nn.init.xavier_normal_(self.cls_token)
        nn.init.xavier_normal_(self.pos_embed)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm_layer = norm_layer or nn.LayerNorm(embed_dim)
        act_layer = act_layer or F.gelu

        self.convlayer = nn.Sequential(
            # nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, ),
            # nn.GELU(),
            # nn.BatchNorm2d(embed_dim),
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim, kernel_size, groups=embed_dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(embed_dim)
                )),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(embed_dim)
            ) for _ in range(convdepth)],
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                embed_dim, num_heads,
                dropout=drop_rate,
                activation=act_layer,
                batch_first=True,
                norm_first=True,
                dim_feedforward=ff_dim,
            ),
            num_layers=depth,
            norm=norm_layer,
        )

    def forward(self, x: torch.Tensor):
        """
            modified from https://github.com/lukas-blecher/LaTeX-OCR/blob/main/pix2tex/models.py
            original LICENSE: MIT
            thanks!
            2022.05.07
        """
        # X: B C H W
        B, _, _, _ = x.shape
        # patch embeding
        x = self.patch_embed(x)  # B D Hp Wp
        _, _, hp, wp = x.shape
        x = self.convlayer(x.to(torch.float))  # b d hp wp
        x = rearrange(x, "b d hp wp -> b (hp wp) d")
        # append class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # position embeding
        assert x.shape[1] <= self.pos_embed.shape[1], f"x.shape: {x.shape}, pos_embedding.shape: {self.pos_embed.shape}"
        # x += self.pos_embed[:, :x.shape[1]]
        pos_emb_idx = repeat(torch.arange(hp)*(self.im_width//self.patch_width-wp), 'hp -> (hp wp)', wp=wp)+torch.arange(hp*wp)
        pos_emb_idx = torch.cat((torch.zeros(1), pos_emb_idx+1), dim=0).long()
        x += self.pos_embed[:, pos_emb_idx]
        x += self.pos_embedding(x)
        x = self.pos_drop(x)

        x = self.transformer(x)

        x = self.norm_layer(x)  # B*(Hp*Wp)*D

        return x


# ConvMixer encoder
class LatexConvMixerEncoder(nn.Module):
    __model_name__ = "convmixer"
    def __init__(self, img_size: Union[Tuple[int, int], int]=224, 
                 patch_size: Union[Tuple[int, int], int]=7,
                 in_chans: int=1, model_dim: int=512, kernel_size: int=7,
                 depth: int=6):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.img_size = to_2tuple(img_size)
        img_check = self.img_size[0] % self.patch_size[0] == 0 and self.img_size[1] % self.patch_size[1] == 0
        assert img_check, f"img_size: {self.img_size}, patch_size: {self.patch_size}"
        num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        self.pos_embedding = PositionalEmbedding(model_dim, num_patches)
        self.norm_layer = nn.LayerNorm(model_dim)
        self.convlayers = nn.Sequential(
            nn.Conv2d(in_chans, model_dim, kernel_size=self.patch_size, stride=self.patch_size, ),
            nn.GELU(),
            nn.BatchNorm2d(model_dim),
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(model_dim, model_dim, kernel_size, groups=model_dim, padding="same"),
                        nn.GELU(),
                        nn.BatchNorm2d(model_dim)
                    )),
                    nn.Conv2d(model_dim, model_dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(model_dim)
                ) for i in range(depth)],
        )

    def pad(self, x: torch.Tensor):
        B, C, H, W = x.shape
        pad_r, pad_b = 0, 0
        ph, pw = self.patch_size
        if H % ph != 0:
            pad_b = (int(H / ph) + 1) * ph - H
        if W % pw != 0:
            pad_r = (int(W / pw) + 1) * pw - W
        if any([pad_r != 0, pad_b != 0]):
            x = F.pad(x, [0, pad_r, 0, pad_b, 0, 0, 0, 0])
        return x

    def forward(self, x):
        x = self.pad(x)
        x = self.convlayers(x.to(torch.float))
        # x.shape: B * model_dim * h/p * w/p
        x = rearrange(x, "b d hp wp -> b (hp wp) d")
        x += self.pos_embedding(x)

        x = self.norm_layer(x)

        return x


class LatexConvNeXtEncoder(nn.Module):
    r"""
        modified from https://github.com/facebookresearch/ConvNeXt, thanks
        LICENSE: MIT
        ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., layer_scale_init_value=1e-6,):
        super(LatexConvNeXtEncoder, self).__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(rearrange(x, "n c h w -> n (h w) c"))
        return x


if __name__ == "__main__":
    # test_img = torch.randn(10, 1, 192, 896)
    test_img = torch.randn(10, 1, 32, 32)
    test_tgt = torch.randint(100, (10, 300))
    # encoder = LatexConvNeXtEncoder(
    #     in_chans=1,
    #     depths=[3, 3, 9, 3],
    #     # dims=[96, 192, 384, 768],
    #     dims=[64, 128, 256, 512],
    # )
    encoder = LatexConvMixerEncoder(
        img_size=(192, 896),
        patch_size=16,
        in_chans=1,
        model_dim=512,
        kernel_size=7,
        depth=4,
    )
    out = encoder(test_img)
    print(out.shape)
