import os
from pathlib import Path
# from pathlib import Path
import random
import torch
import torchvision as tv
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from argparse import ArgumentParser

from .dataset import LatexOCRDataset
from .utils.utils import (dl_collate_pad, post_process, get_latest_ckpt, seed_everything)
from .models.model import build_model, LatexModel
from .metrics import eval_model
from .config import Config
from .utils.dstransforms import get_transforms


def evaluate(model: LatexModel, device, conf: Config, max_batch=0x3f3f3f, sample_freq=5, ):
    val_transforms = get_transforms(conf.min_img_size, conf.max_img_size, mode="val")
    val_ds = LatexOCRDataset(
        conf.dataset_val,
        tex_file = conf.tex_file,
        transforms=val_transforms,
        tokenizer_path=conf.tokenizer_path,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=50,
        shuffle=True,
        num_workers=15,
        pin_memory=True,
        collate_fn=dl_collate_pad,
    )
    print("Evaluating...")
    pred_strs = []
    true_strs = []
    with torch.no_grad():
        val_loss = 0.
        loss_count = 0
        for it, (im, tgt) in enumerate(tqdm(val_loader)):
            xi, xo = tgt[:, :-1].to(device), tgt[:, 1:].to(device)
            logits = model(im.to(device), xi)
            loss = F.cross_entropy(
                logits,
                xo.to(torch.long),
                label_smoothing=conf.label_smooth,
                # ignore_index=conf.pad_token,
            )
            val_loss += loss.item()
            loss_count += 1

            pred = model.generate_batch(im.to(device))
            pred_strs.extend(val_ds.detokenize(pred))
            true_strs.extend(val_ds.detokenize(tgt))

            if (it+1) % sample_freq == 0:
                eval_model(true_strs, pred_strs)
            if it >= max_batch:
                break
    bleu, exact_match, edit_distance = eval_model(true_strs, pred_strs)
    rand_idx = random.randint(0, len(true_strs))
    res = {
        "bleu": bleu, "ematch": exact_match,
        "edist": edit_distance,
        "sample_true": "".join(true_strs[rand_idx]),
        "sample_pred": "".join(pred_strs[rand_idx]),
        "val_loss": val_loss/loss_count,
    }
    return res

def train(data_loader: DataLoader, model: LatexModel, optimizer: optim.Optimizer,
          scheduler: optim.lr_scheduler._LRScheduler, conf: Config,
          tb: SummaryWriter, epoch: int, cmt: str, accum_grd_step: int, device: str,
          eval_freq: int=5000,):
    model.train()
    ds_len = len(data_loader)
    ckpt_name = Path(conf.ckpt_path) / (cmt + f"-ep_{epoch}.pth")
    for it, (img, tgt) in enumerate(tqdm(data_loader)):
        sample_count = epoch*ds_len+it
        # optimizer.zero_grad()
        img = img.to(device)
        xi, xo = tgt[:, :-1].to(device), tgt[:, 1:].to(device)
        logits = model(img, xi)
        loss = F.cross_entropy(
            logits,
            xo.to(torch.long),
            label_smoothing=conf.label_smooth,
            # ignore_index=conf.pad_token,
        )
        if accum_grd_step > 1:
            loss /= accum_grd_step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        # gradient accumulation
        if (it + 1) % accum_grd_step == 0:
            optimizer.step()
            optimizer.zero_grad()
            if optimizer.param_groups[0]['lr'] > conf.lr_min:
                scheduler.step()
            # evaluate
            if (it + 1) % eval_freq == 0:
                model.eval()
                res = evaluate(model, device, conf, max_batch=conf.eval_batch, sample_freq=20)
                tb.add_scalar("eval/loss", res["val_loss"], sample_count)
                tb.add_scalar("eval/bleu4", res["bleu"], sample_count)
                tb.add_scalar("eval/distance", res["edist"], sample_count)
                pred_sample_str = conf.pred_sample_md_template.format(true_sample=res['sample_true'], pred_sample=res['sample_pred'])
                tb.add_text("eval/predict_sample", pred_sample_str, sample_count)
                model.train()

        tb.add_scalar("train/max_memory_alloc", torch.cuda.max_memory_allocated(device)/1000/1000, sample_count)
        tb.add_scalar("train/loss", loss.item(), sample_count)
        tb.add_scalar("lr", optimizer.param_groups[0]['lr'], sample_count)

    torch.save(model.state_dict(), ckpt_name)
    print(f"Saved to {ckpt_name}")

    model.eval()
    res  = evaluate(model, device, conf, max_batch=100, sample_freq=20)
    tb.add_scalar("train/val_bleu4", res["bleu"], epoch)
    tb.add_scalar("train/edit_distance", res["edist"], epoch)
    pred_sample_str = conf.pred_sample_md_template.format(true_sample=res['sample_true'], pred_sample=res['sample_pred'])
    tb.add_text("train/predict_sample", pred_sample_str, epoch)
    print((
            f"epoch: [{epoch}], "
            f"bleu: [{res['bleu']:.4f}]"
        ))
    return res["bleu"]


def main(conf: Config):
    train_transforms = get_transforms(conf.min_img_size, conf.max_img_size, mode="train")
    ds_train = LatexOCRDataset(
        dpath=conf.dataset_train,
        tex_file=conf.tex_file,
        transforms=train_transforms,
        tokenizer_path=conf.tokenizer_path
    )
    train_loader = DataLoader(
        dataset=ds_train,
        batch_size=conf.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=15,
        collate_fn=dl_collate_pad,
    )

    if conf.device:
        device = conf.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
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

    # optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999))
    optimizer = optim.Adam(model.parameters(), lr=conf.lr, betas=conf.betas)
    # optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=int(conf.grad_adj_freq/conf.accum_grd_step),
        gamma=conf.grad_gamma, verbose=False)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=3)

    cmt = str(conf)
    tb = SummaryWriter(comment=cmt, )
    # input_img = torch.randn(10, 1, 192, 896)
    # input_tgt = torch.randint(3, 8000, (10, 300))
    input_img, input_tgt = next(iter(train_loader))
    tb.add_graph(model, (input_img.to(device), input_tgt.to(device)))
    tb.add_image("train/image", tv.utils.make_grid(input_img), 0)

    # latest_ckpt = get_latest_ckpt(conf.ckpt_path, cmt)
    # latest_ckpt = "checkpoints/model-convnext-dsfull-dim64,128,256,512-enc3,3,9,3-dec4-adam-lr0.0001-lsmooth0.0-t0.2-drop0.0-vsize8000-seq512-bs16-accu1-extra_-ep_20.pth"
    # latest_ckpt = conf.checkpoint
    latest_ckpt = ""
    if latest_ckpt:
        print(f"found latest ckpt [{latest_ckpt}], loading....")
        model.load_state_dict(torch.load(latest_ckpt, map_location=device))
    if conf.load_resume:
        if os.path.exists(conf.resume_model):
            model.load_state_dict(torch.load(conf.resume_model, map_location=device))
            print(f"[Resume] Loading model {conf.resume_model}")
        if os.path.exists(conf.resume_optim):
            optimizer.load_state_dict(torch.load(conf.resume_optim, map_location=device))
            print(f"[Resume] Loading optimizer {conf.resume_optim}")
        if os.path.exists(conf.resume_sche):
            scheduler.load_state_dict(torch.load(conf.resume_sche, map_location=device))
            print(f"[Resume] Loading scheduler {conf.resume_sche}")
    # evaluate(model, device, conf, sample_freq=10, max_batch=0x3f3f3f)

    try:
        for epoch in range(conf.epoch_start, conf.epoch_stop):
                train(
                    train_loader, model, optimizer, scheduler,
                    conf,
                    tb, epoch,
                    cmt, conf.accum_grd_step, device,
                    eval_freq=conf.eval_freq,
                )
            # scheduler.step()
    except KeyboardInterrupt:
        # save model
        torch.save(model.state_dict(), conf.resume_model)
        # save optimizer
        torch.save(optimizer.state_dict(), conf.resume_optim)
        # save scheduler
        torch.save(scheduler.state_dict(), conf.resume_sche)

    tb.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", dest="config", type=str, help=".json config file path")

    args = parser.parse_args([
        "-c", "src/config/config_convnext.json",
    ])
    conf = Config(args.config)
    seed_everything(conf.seed)
    main(conf=conf)
