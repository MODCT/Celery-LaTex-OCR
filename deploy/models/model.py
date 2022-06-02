from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import LatexTransformerDecoder
from .encoder import LatexConvNeXtEncoder


__supported_encoder__ = ("vit", "convmixer", "swin", "convnext")


class LatexModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 max_seq_len: int = 512, bos_token: int=1,
                 eos_token: int=2, pad_token: int=0,
                 filter_thres: float=0.9, temperature: float=0.5,
                ):
        super(LatexModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.max_seq_len = max_seq_len
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.filter_thres = filter_thres
        self.temperature = temperature

    @torch.no_grad()
    def generate(self, start_tokens: torch.Tensor, memory: torch.Tensor, seq_len: int=256):
        b, t = start_tokens.shape
        # B, S, E = memory.shape
        out = start_tokens

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(torch.bool).to(x.device)
            # topk and sample
            logits = self.decoder(x, memory, tgt_mask=tgt_mask)[:, -1, :]  # B * vocab_size
            k = int((1 - self.filter_thres) * logits.shape[-1])
            val, idx = torch.topk(logits, k)
            probs = torch.full_like(logits, float("-inf"))
            probs.scatter_(1, idx, val)
            probs = F.softmax(probs/self.temperature, dim=-1)
            sample = torch.multinomial(probs, 1)
            # argmax prob sample
            # probs = F.softmax(logits/self.temperature, dim=-1)
            # sample = probs.argmax(dim=-1).unsqueeze(1)
            # if generate all pad_token, stop
            end_pad = (sample == self.pad_token).all()
            if end_pad:
                sample = torch.ones_like(sample) * self.eos_token
            out = torch.cat((out, sample), dim=-1)
            end_eos = (torch.cumsum(out == self.eos_token, 1)[:, -1] >= 1).all()
            if end_eos:
                break
        out = out[:, t:]
        return out

    @torch.no_grad()
    def generate_batch(self, src: torch.Tensor):
        memory: torch.Tensor = self.encoder(src, )
        # B * 1
        start_tokens = torch.LongTensor([self.bos_token]*src.shape[0]).unsqueeze(1).to(src.device)
        # B * T
        output = self.generate(
            start_tokens=start_tokens,
            memory=memory,
            seq_len=self.max_seq_len,
        )
        return output

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """ autoregressive
            tgt should be target[:, :-1], shape: B * (T-1)
        """
        memory: torch.Tensor = self.encoder(src, )
        # (T-1) * (T-1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).bool().to(tgt.device)
        # tgt_key_padding_mask = (tgt == self.pad_token).to(torch.bool).to(tgt.device)
        # B * (T-1) * num_tokens
        decoded: torch.Tensor = self.decoder(
            tgt, memory, tgt_mask=tgt_mask,
            # tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return decoded.transpose(1, 2)


def build_model(
        in_chans: int=1, model_dim: int=256, num_head: int=16, dropout: float=0.1, 
        dec_depth: int=6, temperature: float=1.,
        vocab_size: int=8000, max_seq_len=512, ff_dim :int=2048,
        next_depths: List[int]=[3, 3, 9, 3], next_dims: List[int]=[64, 128, 256, 512], 
        pdrop_path: float=0.0,
        device: str="cpu",
    ):
    encoder = LatexConvNeXtEncoder(
            in_chans=in_chans,
            depths=next_depths,
            dims=next_dims,
            drop_path_rate=pdrop_path,
    )
    decoder = LatexTransformerDecoder(
        model_dim=model_dim,
        nhead=num_head,
        dropout=dropout,
        depth=dec_depth,
        num_tokens=vocab_size,
        max_seq_len=max_seq_len,
        ff_dim=ff_dim,
    )

    model = LatexModel(
        encoder=encoder,
        decoder=decoder,
        max_seq_len=max_seq_len,
        temperature=temperature,
    ).to(device)

    return model


if __name__ == '__main__':
    ...
