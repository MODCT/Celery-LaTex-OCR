from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .embed import AbsolutePositionalEmbedding


class LatexTransformerDecoder(nn.Module):
    def __init__(self, model_dim: int=786, nhead: int=8, ff_dim: int=2048,
                 dropout: float=0.1, depth: int=8, num_tokens: int=8000,
                 max_seq_len: int=512, norm_eps: float=1e-6, activation=F.gelu,
                 ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, model_dim, padding_idx=0)
        self.pos_emb = AbsolutePositionalEmbedding(max_seq_len, model_dim)
        norm_layer = nn.LayerNorm(model_dim, eps=norm_eps)

        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=model_dim,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            ),
            num_layers=depth,
            norm=norm_layer,
        )

        self.to_logits = nn.Linear(model_dim, num_tokens)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        tgt = self.token_emb(tgt)
        tgt += self.pos_emb(tgt)
        out = self.transformer(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        out = self.to_logits(out)  # B * num_tokens
        return out
