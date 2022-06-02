from pathlib import Path


class Config(object):
    #%%% general and dataset %%%#
    seed = 241
    min_img_size = (32, 32)
    max_img_size = (192, 896)
    checkpoint = "deploy/checkpoint/checkpoint.pth"
    tokenizer = "dataset/data/tokenizer.json"
    # device = "cuda"
    device = ""
    ###   special tokens   ###
    pad_token = 0
    bos_token = 1
    eos_token = 2

    model_name = "convnext"

    #%%%%%% hyper parameters %%%%%#
    ###   optimizer   ###
    lr = 1e-4
    lr_min = 1e-7
    betas = (0.9, 0.999)
    batch_size = 16
    accum_grd_step = 1
    ###   model encoder/decoder   ###
    mdim = 512
    nhead = 8
    in_chans = 1
    pdrop = 0.0
    dec_depth = 4
    ff_dim = 2048
    vocab_size = 8000
    max_seq = 512
    label_smooth = 0.0
    temperature = 0.2
    ###   convnext   ###
    next_depths = [3, 3, 9, 3]
    next_dims = [64, 128, 256, 512]  # last dim must be equal to `mdim`
    # next_dims = [96, 192, 384, 768]
    pdrop_path = 0.0


if __name__ == "__main__":
    ...