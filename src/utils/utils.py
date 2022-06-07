import os
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
import torch
import re
import torchvision.transforms.functional as VF
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer
from tokenizers.models import BPE

_buckets = [32, 64, 96, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896]


def dl_collate_pad(batch):
    # B C H W
    data: List[torch.Tensor] = [b[0] for b in batch]
    # [B * tokenizers.Encoding]
    tgt: List[torch.Tensor] = [torch.Tensor(b[1]).to(torch.int32) for b in batch]
    del batch
    # pad tgt
    # B * max(t.num_tokens)
    tgt: torch.Tensor = pad_sequence(tgt, batch_first=True, padding_value=0)
    # pad data
    buckets = [[i, j] for i in _buckets for j in _buckets]
    max_dim_h, max_dim_w = get_max_hw(data)
    # max_dim_h = max([d.shape[-2] for d in data])
    # max_dim_w = max([d.shape[-1] for d in data])
    # print(max_dim_h, max_dim_w)
    new_size = get_new_size((max_dim_h, max_dim_w), buckets=buckets)
    data = torch.cat(
        [
            # VF.pad(d, [0, 0, max_dim_w-d.shape[-1], max_dim_h-d.shape[-2]], 0) 
            VF.pad(d, [0, 0, new_size[1]-d.shape[-1], new_size[0]-d.shape[-2]], 0) 
            for d in data
        ],
        dim = 0,
    ).unsqueeze(1)

    return [data, tgt]


def get_max_hw(data: List[torch.Tensor]):
    mh, mw = 0, 0
    for d in data:
        if d.shape[-2] > mh:
            mh = d.shape[-2]
        if d.shape[-1] > mw:
            mw = d.shape[-1]
    return (mh, mw)


def get_new_size(old_size, buckets):
    """Computes new size from buckets
    from https://github.com/LinXueyuanStdio/LaTeX_OCR_PRO
    Args:
        old_size: (width, height) or (height, width)
        buckets: list of sizes
    Returns:
        new_size: shape same as old_size,
            original size or first bucket in iter order that matches the size.
    """
    if buckets is None:
        return old_size
    else:
        d1, d2 = old_size
        for (d1_b, d2_b) in buckets:
            if d1_b >= d1 and d2_b >= d2:
                return d1_b, d2_b
        # no match, return max (last one)
        return buckets[-1]


def to_2tuple(t: Union[Tuple, int]):
    return t if isinstance(t, tuple) else (t, t)


def detokenize(tokens, tokenizer: Tokenizer):
    if isinstance(tokens, list):
        tokens = torch.Tensor(tokens).to(torch.int)
    if len(tokens.shape) == 1:
        tokens = tokens.unsqueeze(0)
    toks = [[tokenizer.id_to_token(t) for t in tok] for tok in tokens.tolist()] # type: ignore
    for b in range(len(toks)):
        for i in reversed(range(len(toks[b]))):
            if toks[b][i] is None:
                toks[b][i] = ''
            toks[b][i] = toks[b][i].replace('Ġ', ' ').replace("Ċ", "").strip()
            if toks[b][i] in ('[BOS]', '[EOS]', '[PAD]'):
                del toks[b][i]
    return toks


def token2str(tokens: Union[torch.Tensor, List], tokenizer) -> list:
        if isinstance(tokens, list):
            tokens = torch.LongTensor(tokens)
        if len(tokens.shape) == 1:
            tokens = tokens.unsqueeze(0)
        dec = [tokenizer.decode(ids=tok, skip_special_tokens=False) for tok in tokens.tolist()]
        return [
            ''.join(detok.split(' '))
            .replace('Ġ', ' ') # space
            .replace("Ċ", "")  # newline
            .replace('[EOS]', '')
            .replace('[BOS]', '')
            .replace('[PAD]', '')
            .strip() for detok in dec
        ]


def get_tokenizer(tpath: str):
    assert os.path.exists(tpath), f"tokenizer file {tpath} not exist!"
    tokenizer = Tokenizer(BPE())
    tokenizer = tokenizer.from_file(tpath)
    return tokenizer


def seed_everything(seed: int):
    """
        from https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
    """
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def post_process(s: str):
    """
        modified from https://github.com/lukas-blecher/LaTeX-OCR, thanks!
        Remove unnecessary whitespace from LaTeX code.
        Args:
            s (str): Input string
        Returns:
            str: Processed latex string
    """
    text_reg = r'(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})'
    letter = '[a-zA-Z]'
    noletter = '[\W_^\d]'
    names = [x[0].replace(' ', '') for x in re.findall(text_reg, s)]
    s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
    news = s
    while True:
        s = news
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, noletter), r'\1\2', s)
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, letter), r'\1\2', news)
        news = re.sub(r'(%s)\s+?(%s)' % (letter, noletter), r'\1\2', news)
        if news == s:
            break
    return s


def get_latest_ckpt(ckpt_path: Union[str, Path], cmt: str="", ext="pth"):
    ckpt_path = Path(ckpt_path)
    ckpts = list(ckpt_path.glob(f"{cmt}*.{ext}"))
    if len(ckpts) == 0:
        return None
    mtime = [os.path.getmtime(p) for p in ckpts]
    newest_idx = np.argmax(mtime)
    return ckpts[newest_idx]


def get_model_param_size(model):
    return sum(p.numel() for p in model.parameters()) / 1024 / 1024


if __name__ == '__main__':
    ...
