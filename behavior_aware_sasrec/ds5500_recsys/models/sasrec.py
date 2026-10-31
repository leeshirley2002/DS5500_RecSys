"""SASRec and behavior-aware SASRec (Kang & McAuley, ICDM 2018 + cart behaviors)."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, H] -> conv expects [B, H, L]
        out = x.transpose(1, 2)
        out = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(out)))))
        out = out.transpose(1, 2)
        return out


class SASRecBlock(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = PointWiseFeedForward(hidden_size, dropout)
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-8)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-8)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(
            h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + self.dropout(attn_out)
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


class SASRec(nn.Module):
    """Purchase-only SASRec baseline (item + position embeddings)."""

    def __init__(
        self,
        n_items: int,
        hidden_size: int = 64,
        max_len: int = 50,
        n_blocks: int = 2,
        n_heads: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_items = n_items
        self.hidden_size = hidden_size
        self.max_len = max_len

        self.item_emb = nn.Embedding(n_items + 1, hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_size)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [SASRecBlock(hidden_size, n_heads, dropout) for _ in range(n_blocks)]
        )
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        self._init_weights()

    def _init_weights(self) -> None:
        std = 0.02
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
            if isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # True means "do not attend" for nn.MultiheadAttention.
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def encode(
        self,
        item_ids: torch.Tensor,
        behavior_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return contextualized sequence embeddings [B, L, H]."""
        del behavior_ids  # unused in purchase-only model
        batch_size, seq_len = item_ids.shape
        positions = torch.arange(seq_len, device=item_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.item_emb(item_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        pad_mask = item_ids == 0  # [B, L]
        attn_mask = self._causal_mask(seq_len, item_ids.device)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, key_padding_mask=pad_mask)
        return self.final_norm(x)

    def forward(
        self,
        item_ids: torch.Tensor,
        behavior_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Logits over all items using the last non-pad position... actually full seq.

        Returns [B, L, n_items+1] scores against shared item embedding table
        (padding column 0 is ignored in loss).
        """
        seq_emb = self.encode(item_ids, behavior_ids)
        # Tie weights with item embedding matrix.
        scores = torch.matmul(seq_emb, self.item_emb.weight.transpose(0, 1))
        return scores

    def predict(
        self,
        item_ids: torch.Tensor,
        behavior_ids: Optional[torch.Tensor] = None,
        candidate_items: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score candidates (or all items) from the last valid history position.

        item_ids: [B, L]
        candidate_items: [B, C] optional
        returns: [B, C] or [B, n_items+1]
        """
        seq_emb = self.encode(item_ids, behavior_ids)
        # Gather embedding at last non-padding index.
        lengths = (item_ids != 0).sum(dim=1).clamp(min=1)  # [B]
        last_idx = lengths - 1
        final = seq_emb[torch.arange(seq_emb.size(0), device=seq_emb.device), last_idx]  # [B, H]

        if candidate_items is None:
            return torch.matmul(final, self.item_emb.weight.transpose(0, 1))

        cand_emb = self.item_emb(candidate_items)  # [B, C, H]
        return (cand_emb * final.unsqueeze(1)).sum(dim=-1)


class BehaviorAwareSASRec(SASRec):
    """SASRec with explicit behavior-type embeddings (buy / add / remove).

    Same transformer backbone as purchase-only SASRec; the only added signal
    is a behavior embedding summed into the input, isolating the effect of
    cart events from architecture changes.
    """

    def __init__(
        self,
        n_items: int,
        n_behaviors: int = 3,
        hidden_size: int = 64,
        max_len: int = 50,
        n_blocks: int = 2,
        n_heads: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(
            n_items=n_items,
            hidden_size=hidden_size,
            max_len=max_len,
            n_blocks=n_blocks,
            n_heads=n_heads,
            dropout=dropout,
        )
        # behavior ids: 0 pad, 1 buy, 2 add, 3 remove  => n_behaviors+1 embeddings
        self.behavior_emb = nn.Embedding(n_behaviors + 1, hidden_size, padding_idx=0)
        nn.init.normal_(self.behavior_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.behavior_emb.weight[0].fill_(0)

    def encode(
        self,
        item_ids: torch.Tensor,
        behavior_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if behavior_ids is None:
            raise ValueError("BehaviorAwareSASRec requires behavior_ids.")
        batch_size, seq_len = item_ids.shape
        positions = torch.arange(seq_len, device=item_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.item_emb(item_ids) + self.behavior_emb(behavior_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        pad_mask = item_ids == 0
        attn_mask = self._causal_mask(seq_len, item_ids.device)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, key_padding_mask=pad_mask)
        return self.final_norm(x)


def bce_ranking_loss(
    model: SASRec,
    item_ids: torch.Tensor,
    behavior_ids: torch.Tensor,
    pos_items: torch.Tensor,
    neg_items: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy on positive/negative items from final history state."""
    seq_emb = model.encode(item_ids, behavior_ids)
    lengths = (item_ids != 0).sum(dim=1).clamp(min=1)
    last_idx = lengths - 1
    final = seq_emb[torch.arange(seq_emb.size(0), device=seq_emb.device), last_idx]

    pos_emb = model.item_emb(pos_items)
    pos_logits = (final * pos_emb).sum(dim=-1)
    neg_emb = model.item_emb(neg_items)  # [B, N, H] or [B, H]
    if neg_emb.dim() == 2:
        neg_logits = (final * neg_emb).sum(dim=-1)
        loss = -F.logsigmoid(pos_logits).mean() - F.logsigmoid(-neg_logits).mean()
    else:
        neg_logits = (neg_emb * final.unsqueeze(1)).sum(dim=-1)
        loss = -F.logsigmoid(pos_logits).mean() - F.logsigmoid(-neg_logits).mean()
    return loss
