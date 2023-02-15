import re
from argparse import ArgumentParser
from typing import Union, List
import time

import numpy as np
from numpy.typing import NDArray
# from viztracer import VizTracer
import torch
import torchvision.transforms.functional as VF
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import BPE

from .config import Config
from .models.model import build_model
from .utils.dstransforms import get_transforms
from .utils.utils import post_process, seed_everything


def token2str(tokenizer: Tokenizer, tokens):
    # tokens: (B, N)
    dec = [tokenizer.decode(ids=t) for t in tokens]
    dec = [
            ''.join(detok.split(' '))
            .replace('Ġ', ' ') # space
            .replace("Ċ", "")  # newline
            .replace('[EOS]', '')
            .replace('[BOS]', '')
            .replace('[PAD]', '')
            .strip() for detok in dec
        ]
    # (B, BW)
    return dec

def post_process(s: str) -> str:
    """
        modified from https://github.com/lukas-blecher/LaTeX-OCR, thanks!
        Remove unnecessary whitespace from LaTeX code.
    Args:
        s (str): Input string
    Returns:
        str: Processed latex string
    """
    text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
    letter = "[a-zA-Z]"
    noletter = "[\W_^\d]"
    names = [x[0].replace(" ", "") for x in re.findall(text_reg, s)]
    s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
    news = s
    while True:
        s = news
        news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", s)
        news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
        news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
        if news == s:
            break
    return s


def main(impath: str, conf: Config):
    model = build_model(
        in_chans=conf.in_chans,
        model_dim=conf.mdim,
        num_head=conf.nhead,
        dropout=conf.pdrop,
        dec_depth=conf.dec_depth,
        ff_dim = conf.ff_dim,
        vocab_size=conf.vocab_size,
        max_seq_len=conf.max_seq,
        temperature=conf.temperature,
        next_depths=conf.next_depths,
        next_dims=conf.next_dims,
        pdrop_path=conf.pdrop_path,
        model_name=conf.model_name,
        device="cpu",
    )
    model.load_state_dict(torch.load(conf.checkpoint, map_location="cpu"))
    model.eval()

    tokenizer = Tokenizer(BPE()).from_file(conf.tokenizer_path)

    img = Image.open(impath, ).convert("L")  # H W
    img = torch.from_numpy(np.array(img, dtype=np.float32)).unsqueeze(0)
    transforms = get_transforms(conf.min_img_size, conf.max_img_size, "deploy")
    img = transforms(img)
    # with VizTracer(output_file="./predict.html") as viztracer:
    #     res = model([img])
    s0 = time.time()
    res = model(img.unsqueeze(0), generate=True)
    s1 = time.time()
    res_str = token2str(tokenizer=tokenizer, tokens=res.tolist())
    res_str = [post_process(r) for r in res_str]
    print(res_str)
    print(f"Inference took {s1-s0}s")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", dest="image", type=str, help="image to predict")
    parser.add_argument("-c", dest="config", type=str, help=".json config file")
    args = parser.parse_args([
        "-i", "tmp/218395745-8e87de7d-522c-4268-84e0-665112b93635.jpg",
        "-c", "src/config/config_convnext.json",
    ])
    seed_everything(241)
    main(args.image, Config(args.config))
