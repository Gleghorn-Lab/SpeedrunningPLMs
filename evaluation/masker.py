"""Standardized protein masked-language-model corruption."""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ProteinMasker(nn.Module):
    def __init__(self, tokenizer, mask_rate: float = 0.15):
        super().__init__()
        self.mask_token_id = tokenizer.mask_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.mask_rate = mask_rate

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return masked input IDs and labels with unmasked positions ignored."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        mask_probabilities = torch.full(
            (batch_size, seq_len),
            self.mask_rate,
            device=device,
        )
        mask_indices = torch.rand(batch_size, seq_len, device=device) < mask_probabilities

        cls_mask = input_ids == self.cls_token_id
        eos_mask = input_ids == self.eos_token_id
        mask_indices = mask_indices & ~cls_mask & ~eos_mask & attention_mask.bool()

        # Avoid empty-label batches for short sequences and small batch sizes.
        for row in range(batch_size):
            if not mask_indices[row].any() and attention_mask[row].sum() > 2:
                valid_positions = (
                    ~cls_mask[row]
                    & ~eos_mask[row]
                    & attention_mask[row].bool()
                )
                if valid_positions.any():
                    candidates = valid_positions.nonzero(as_tuple=True)[0]
                    selected = candidates[
                        torch.randint(candidates.numel(), (1,), device=device)
                    ]
                    mask_indices[row, selected] = True

        masked_input_ids = torch.where(mask_indices, self.mask_token_id, input_ids)
        labels = input_ids.clone()
        labels[~mask_indices | (attention_mask == 0)] = -100
        return masked_input_ids, labels
