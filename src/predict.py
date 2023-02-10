# from pathlib import Path
import numpy as np
# from viztracer import VizTracer
import torch
import torchvision.transforms.functional as VF
from argparse import ArgumentParser
from PIL import Image

from .models.model import build_model
from .config import Config
from .utils.dstransforms import get_transforms
from .utils.utils import post_process, seed_everything


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

    img = Image.open(impath, ).convert("L")  # H W
    img = torch.from_numpy(np.array(img, dtype=np.float32)).unsqueeze(0)
    # img = np.array(img, dtype=np.float32)[np.newaxis, ...]
    transforms = get_transforms(conf.min_img_size, conf.max_img_size, "deploy")
    img = transforms(img)
    np.savetxt("src.txt", img[0].tolist(), )
    # with VizTracer(output_file="./predict.html") as viztracer:
    #     res = model([img])
    res = model(img.unsqueeze(0), generate=True)
    print(res[0])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", dest="image", type=str, help="image to predict")
    parser.add_argument("-c", dest="config", type=str, help=".json config file")
    args = parser.parse_args([
        "-i", "tmp/0.png",
        "-c", "src/config/config_convnext.json",
    ])
    seed_everything(241)
    main(args.image, Config(args.config))
