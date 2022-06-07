# from pathlib import Path
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import LatexOCRDataset
from .utils.utils import (dl_collate_pad, post_process, get_latest_ckpt, seed_everything)
from .models.model import build_model, LatexModel
from .metrics import eval_model
from .config import Config
from .utils.dstransforms import get_transforms


def evaluate(model: LatexModel, device, val_ds: LatexOCRDataset, val_loader, max_batch=0x3f3f3f, sample_freq=20,):
    print("Evaluating...")
    pred_strs = []
    true_strs = []
    model.eval()
    with torch.no_grad():
        for it, (im, tgt) in enumerate(tqdm(val_loader)):
            pred = model.generate_batch(im.to(device))
            pred_strs.extend(val_ds.detokenize(pred))
            true_strs.extend(val_ds.detokenize(tgt))

            if (it+1) % sample_freq == 0:
                eval_model(true_strs, pred_strs)
            if it >= max_batch:
                break
    print("ALL DATASET: ")
    bleu, exact_match, edit_distance = eval_model(true_strs, pred_strs)


def main(conf: Config):
    val_transforms = get_transforms(conf.min_img_size, conf.max_img_size, "val")
    val_ds = LatexOCRDataset(
        conf.dataset_val,
        tex_file=conf.tex_file,
        transforms=val_transforms,
        # transforms=deploy_transforms,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=True,
        num_workers=15,
        pin_memory=True,
        collate_fn=dl_collate_pad,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    model = build_model(
        img_size=conf.max_img_size,
        patch_size=conf.psize,
        in_chans=conf.in_chans,
        model_dim=conf.mdim,
        num_head=conf.nhead,
        dropout=conf.pdrop,
        enc_depth=conf.enc_depth,
        enc_convdepth=conf.enc_convdepth,
        dec_depth=conf.dec_depth,
        vocab_size=conf.vocab_size,
        max_seq_len=conf.max_seq,
        temperature=conf.temperature,
        kernel_size=conf.kernel_size,
        model_name=conf.model_name,
        next_depths=conf.next_depths,
        next_dims=conf.next_dims,
        pdrop_path=conf.pdrop_path,
        device=device,
    )
    try:
        print(f"Loading checkpoint {conf.checkpoint}")
        model.load_state_dict(torch.load(conf.checkpoint, map_location=device))
    except Exception as e:
        print(f"Load checkpoint failed, {e}")
    evaluate(model, device, val_ds, val_loader, sample_freq=20)

 
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", dest="config", type=str, help=".json config file path")

    args = parser.parse_args([
        "-c", "src/config/config_convnext.json",
    ])
    conf = Config(args.config)
    seed_everything(conf.seed)
    main(conf=conf)
