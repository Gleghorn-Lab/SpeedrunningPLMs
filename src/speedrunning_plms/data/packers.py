from dataclasses import dataclass
from typing import Iterable, List

import torch


@dataclass(frozen=True)
class ChunkPacker:
    max_length: int
    eos_token_id: int
    pad_token_id: int

    def pack(self, raw_tokens: torch.Tensor) -> Iterable[torch.Tensor]:
        eos_positions = (raw_tokens == self.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) == 0:
            return

        chunk_parts: List[torch.Tensor] = []
        chunk_len = 0

        prev_start = 0
        for i in range(len(eos_positions)):
            curr_eos = eos_positions[i].item()
            doc = raw_tokens[prev_start:curr_eos + 1]
            prev_start = curr_eos + 1
            doc_len = len(doc)

            if doc_len > self.max_length:
                if chunk_len > 0:
                    padding = torch.full((self.max_length - chunk_len,), self.pad_token_id, dtype=torch.uint8)
                    yield torch.cat(chunk_parts + [padding])
                    chunk_parts = []
                    chunk_len = 0
                yield doc[:self.max_length].clone()
                continue

            if doc_len + chunk_len > self.max_length:
                padding = torch.full((self.max_length - chunk_len,), self.pad_token_id, dtype=torch.uint8)
                yield torch.cat(chunk_parts + [padding])
                chunk_parts = []
                chunk_len = 0

            chunk_parts.append(doc)
            chunk_len += doc_len

            if chunk_len == self.max_length:
                yield torch.cat(chunk_parts)
                chunk_parts = []
                chunk_len = 0

        if chunk_len > 0:
            padding = torch.full((self.max_length - chunk_len,), self.pad_token_id, dtype=torch.uint8)
            yield torch.cat(chunk_parts + [padding])


@dataclass(frozen=True)
class LegacyFlatPacker:
    seq_len: int
    eos_token_id: int
    pad_token_id: int

    def split_oversized(self, sample: torch.Tensor) -> Iterable[torch.Tensor]:
        for j in range(0, len(sample), self.seq_len):
            chunk = sample[j:j + self.seq_len]
            if len(chunk) < self.seq_len:
                padding = torch.full((self.seq_len - len(chunk),), self.pad_token_id, dtype=torch.uint8)
                chunk = torch.cat([chunk, padding])
            yield chunk
