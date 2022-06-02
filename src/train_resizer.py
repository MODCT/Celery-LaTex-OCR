import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm.auto import tqdm
from typing import Tuple
from torch.utils.tensorboard.writer import SummaryWriter

from .utils.utils import dl_collate_resizer
from .config import Config
from .dataset import LatexOCRDataset
from .dstransforms import train_transforms, val_transforms

CONF = Config()
tb = SummaryWriter(comment="resizer", )
device = "cpu"

model = nn.Transformer(
    d_model=2, nhead=2, num_encoder_layers=2, num_decoder_layers=2,
    dropout=0.1, batch_first=True, dim_feedforward=512,
).to(device)


def train():
    train_ds = LatexOCRDataset(
        CONF.dataset_train,
        tex_file=CONF.tex_file,
        transforms=train_transforms,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=50,
        shuffle=True,
        num_workers=15,
        pin_memory=True,
        collate_fn=dl_collate_resizer,
    )
    optimizer = Adam(model.parameters(), 0.0001)
    all_imshape = []
    all_impad = []
    for i, (imsize, impad) in tqdm(enumerate(train_loader)):
        all_imshape.extend(imsize.tolist())
        all_impad.extend(impad.tolist())
    import json
    with open("imshape.json", "w") as f:
        json.dump(all_imshape, f)
    with open("impad.json", "w") as f:
        json.dump(all_impad, f)

    # for epoch in range(100):
        # for i, (imsize, impad) in tqdm(enumerate(train_loader)):
        #     imsize, impad = imsize.to(device), impad.to(device)
        #     sample_count = epoch * len(train_loader) + i
        #     optimizer.zero_grad()
        #     logits = model(imsize.to(torch.float), impad.to(torch.float))
        #     loss = F.cross_entropy(logits, impad)
        #     loss.backward()
        #     optimizer.step()
        #     tb.add_scalar("train/loss", loss.item(), sample_count)


def main():
    train()

tb.close()

if __name__ == "__main__":
    main()
