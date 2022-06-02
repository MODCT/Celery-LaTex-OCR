# from pathlib import Path
import torch
import torchvision as tv

from .models.model import build_model, LatexModel
from .config import Config
from .dstransforms import deploy_transforms
from .utils.utils import get_tokenizer, token2str

CONF = Config()
tokenizer = get_tokenizer(CONF.tokenizer)

def predict(model: LatexModel, img, device="cpu"):
    model.eval()
    with torch.no_grad():
        pred = model.generate_batch(img.to(device))
    pred = token2str(pred, tokenizer=tokenizer)
    return pred

def main():
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    model = build_model(
        in_chans=CONF.in_chans,
        model_dim=CONF.mdim,
        num_head=CONF.nhead,
        dropout=CONF.pdrop,
        dec_depth=CONF.dec_depth,
        ff_dim = CONF.ff_dim,
        vocab_size=CONF.vocab_size,
        max_seq_len=CONF.max_seq,
        temperature=CONF.temperature,
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
    
    impath = "tmp/239347.png"
    img = img = tv.io.read_image(impath, mode=tv.io.ImageReadMode.GRAY)
    img = deploy_transforms(img.to(torch.float)).unsqueeze(0)
    res = predict(model, img, device=device)
    print(res[0].replace("\\\\", "\\"))


if __name__ == "__main__":
    main()
