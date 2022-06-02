# from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import LatexOCRDataset
from .utils.utils import (dl_collate_pad, post_process, get_latest_ckpt, seed_everything)
from .models.model import build_model, LatexModel
from .metrics import eval_model
from .config import Config
from .dstransforms import val_transforms, deploy_transforms

CONF = Config()

def evaluate(model: LatexModel, device, val_ds, val_loader, max_batch=0x3f3f3f, sample_freq=20,):
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


def main():
    val_ds = LatexOCRDataset(
        CONF.dataset_val,
        tex_file=CONF.tex_file,
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
        img_size=CONF.max_img_size,
        patch_size=CONF.psize,
        in_chans=CONF.in_chans,
        model_dim=CONF.mdim,
        num_head=CONF.nhead,
        dropout=CONF.pdrop,
        enc_depth=CONF.enc_depth,
        enc_convdepth=CONF.enc_convdepth,
        dec_depth=CONF.dec_depth,
        vocab_size=CONF.vocab_size,
        max_seq_len=CONF.max_seq,
        temperature=CONF.temperature,
        kernel_size=CONF.kernel_size,
        model_name=CONF.model_name,
        next_depths=CONF.next_depths,
        next_dims=CONF.next_dims,
        pdrop_path=CONF.pdrop_path,
        device=device,
    )
    try:
        print(f"Loading checkpoint {CONF.checkpoint}")
        model.load_state_dict(torch.load(CONF.checkpoint, map_location=device))
    except Exception as e:
        print(f"Load checkpoint failed, {e}")
    evaluate(model, device, val_ds, val_loader, sample_freq=20)

 
if __name__ == "__main__":
    main()
