from pathlib import Path
from typing import List, Union
import torch
import torch.nn as nn
import torchvision as tv
# import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# import numpy as np
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from .config import Config

from .utils.customtransforms import PadMaxResize
from .utils.utils import dl_collate_pad


class LatexOCRDataset(Dataset):
    img_path: Union[str, Path, None] = None
    tex_strs: Union[str, None] = None
    _len_: int = 0
    _default_tokenizer_ = "dataset/data/tokenizer.json"

    def __init__(
            self, dpath: Union[str, Path],
            tex_file: Union[str, Path],
            transforms: nn.Module = None,
            imgfmt: str = "png",
            tokenizer_path: str = None,
            bos_token: int = 1,
            eos_token: int = 2,):
        super().__init__()
        if isinstance(dpath, str):
            dpath = Path(dpath)
        if isinstance(tex_file, str):
            tex_file = Path(tex_file)
        self.dpath = dpath
        self.tex_file = tex_file
        assert self.dpath.exists() and self.tex_file.exists(), f"dataset {self.dpath} or tex file is not exists!"
        self.imgfmt = imgfmt
        self.transforms = transforms
        self.tokenizer = self.load_tokenizer(tokenizer_path)
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.init_dataset()

    def load_tokenizer(self, path: str, inplace=False):
        path = path or self._default_tokenizer_
        # print(f"Using tokenizer {path}")
        tokenizer = Tokenizer(BPE())
        tokenizer = tokenizer.from_file(path)
        if inplace:
            self.tokenizer = tokenizer
        return tokenizer

    def detokenize(self, tokens: Union[torch.Tensor, List]):
        if isinstance(tokens, list):
            tokens = torch.Tensor(tokens).to(torch.int)
        if len(tokens.shape) == 1:
            tokens = tokens.unsqueeze(0)
        toks = [[self.tokenizer.id_to_token(t) for t in tok] for tok in tokens.tolist()] # type: ignore
        for b in range(len(toks)):
            for i in reversed(range(len(toks[b]))):
                if toks[b][i] is None:
                    toks[b][i] = ''
                toks[b][i] = toks[b][i].replace('Ġ', ' ').replace("Ċ", "").strip()
                if toks[b][i] in ('[BOS]', '[EOS]', '[PAD]'):
                    del toks[b][i]
        return toks

    def token2str(self, tokens: Union[torch.Tensor, List]) -> list:
        if isinstance(tokens, list):
            tokens = torch.LongTensor(tokens)
        if len(tokens.shape) == 1:
            tokens = tokens.unsqueeze(0)
        dec = [self.tokenizer.decode(ids=tok, skip_special_tokens=False) for tok in tokens.tolist()]
        return [
            ''.join(detok.split(' '))
            .replace('Ġ', ' ') # space
            .replace("Ċ", "")  # newline
            .replace('[EOS]', '')
            .replace('[BOS]', '')
            .replace('[PAD]', '')
            .strip() for detok in dec
        ]

    def init_dataset(self):
        img_path = [s for s in self.dpath.glob(f"**/*.{self.imgfmt}")]
        with open(self.tex_file, "r") as f:
            tex_list = f.readlines()
        assert len(tex_list) >= len(img_path), "number of tex string < images"
        tex_idxs = [int(s.name.split(".")[0]) for s in img_path]
        self.tex_strs = [tex_list[i] for i in tex_idxs]
        self.img_path = [str(p.absolute()) for p in img_path]
        self._len_ = len(self.img_path)

    def __getitem__(self, idx):
        img = tv.io.read_image(self.img_path[idx], mode=tv.io.ImageReadMode.GRAY)
        tex = self.tokenizer.encode(self.tex_strs[idx])
        tex_ids = [self.bos_token, *tex.ids, self.eos_token]
        # tex_atnmsk = [1, *tex.attention_mask, 1]
        if self.transforms is not None:
            img = self.transforms(img.to(torch.float16))
        # return img, (tex_ids, tex_atnmsk)
        return img, tex_ids

    def __len__(self):
        return self._len_


def generate_tokenizer(equations: List[str], output: str, vocab_size: int):
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(special_tokens=["[PAD]", "[BOS]", "[EOS]"], vocab_size=vocab_size, show_progress=True)
    tokenizer.train(equations, trainer)
    tokenizer.save(path=output, pretty=False)


if __name__ == "__main__":
    ...
