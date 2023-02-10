import json
from pathlib import Path
from typing import Dict, Sequence, Union


class Config(object):
    ##%%% general and dataset %%%##
    seed = 0
    min_img_size = (32, 32)
    max_img_size = (192, 896)
    dataset_train = ""
    dataset_val = ""
    tex_file = ""
    resizer = ""
    ckpt_path = ""
    device = ""
    ###   special tokens   ###
    pad_token = 0
    bos_token = 1
    eos_token = 2

    # model_name = "vit"
    # model_name = "convmixer"
    model_name = ""
    ds = ""

    ##%%%%%% hyper parameters %%%%%##
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
    enc_depth = 4
    dec_depth = 4
    enc_convdepth = 16
    pdrop = 0.0
    ff_dim = 2048
    vocab_size = 8000
    max_seq = 512
    label_smooth = 0.0
    psize = 16  # vit
    temperature = 0.2
    kernel_size = 7  # convmixer
    # psize = 7  # convmixer
    ###   convnext   ###
    next_depths = [3, 3, 9, 3]
    next_dims = [64, 128, 256, 512]  # last dim must be equal to `mdim`
    pdrop_path = 0.0

    ##%%%%% train and eval control %%%%%##
    epoch_start = 0
    epoch_stop = 100
    grad_adj_freq = 10000
    grad_gamma = 0.98
    eval_freq = 5000
    eval_batch = 30
    load_resume = False
    resume_model = ""
    resume_optim = ""
    resume_sche = ""
    checkpoint = ""
    tokenizer_path = ""

    ##%%%%% extra info %%%%%##
    extra = ""
    pred_sample_md_template = """|true|pred|\n|-|-|\n|{true_sample}|{pred_sample}|"""

    def __init__(self, conf_path: str):
        self.load(conf_path)

    def load(self, conf_path: str):
        with open(conf_path, 'r', encoding="utf-8") as f:
            conf: Dict[str, Union[str, int]] = json.load(f)
        for k, v in conf.items():
            setattr(self, k, v)

    def __str__(self):
        if self.model_name == "vit":
            info = (
                f"model-{self.model_name}-ds{self.ds}-dim{self.mdim}-ps{self.psize}-ks{self.kernel_size}"
                f"-enc{self.enc_depth}-dec{self.dec_depth}-encconv{self.enc_convdepth}"
                f"-adam-lr{self.lr}-lsmooth{self.label_smooth}"
                f"-t{self.temperature}"
                f"-drop{self.pdrop}-vsize{self.vocab_size}-seq{self.max_seq}"
                f"-bs{self.batch_size}-accu{self.accum_grd_step}"
                f"-extra_{self.extra}"
            )
        elif self.model_name == "convnext":
            sdim = ','.join([str(i) for i in self.next_dims])
            senc = ','.join([str(i) for i in self.next_depths])
            info = (
                f"model-{self.model_name}-ds{self.ds}-dim{sdim}"
                f"-enc{senc}-dec{self.dec_depth}"
                f"-adam-lr{self.lr}-lsmooth{self.label_smooth}"
                f"-t{self.temperature}"
                f"-drop{self.pdrop}-vsize{self.vocab_size}-seq{self.max_seq}"
                f"-bs{self.batch_size}-accu{self.accum_grd_step}"
                f"-extra_{self.extra}"
            )
        elif self.model_name == "convmixer":
            info = (
                f"model-{self.model_name}-ds{self.ds}-dim{self.mdim}-ps{self.psize}-ks{self.kernel_size}"
                f"-enc{self.enc_depth}-dec{self.dec_depth}"
                f"-adam-lr{self.lr}-lsmooth{self.label_smooth}"
                f"-t{self.temperature}"
                f"-drop{self.pdrop}-vsize{self.vocab_size}-seq{self.max_seq}"
                f"-bs{self.batch_size}-accu{self.accum_grd_step}"
                f"-extra_{self.extra}"
            )
        return info


if __name__ == "__main__":
    print(str(Config()))
