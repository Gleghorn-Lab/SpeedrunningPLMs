import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speedrunning_plms.data import TokenIds, read_shard_num_tokens, read_shard_tokens, write_shard
from speedrunning_plms.data.loaders import ChunkedTrainDataset, EvalLoader


TOKEN_IDS = TokenIds(cls_token_id=0, eos_token_id=2, pad_token_id=1, mask_token_id=32)


class DataContractTests(unittest.TestCase):
    def test_shard_round_trip_preserves_header_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tiny.bin"
            tokens = np.array([0, 5, 2, 0, 6, 2], dtype=np.uint8)
            write_shard(path, tokens)

            self.assertEqual(read_shard_num_tokens(path), len(tokens))
            torch.testing.assert_close(read_shard_tokens(path), torch.tensor(tokens, dtype=torch.uint8))

    def test_eval_loader_accepts_token_ids_and_yields_cpu_masked_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            os.chdir(tmpdir)
            try:
                data_dir = Path("data")
                data_dir.mkdir()
                write_shard(data_dir / "tiny_valid_000000.bin", np.array([0, 5, 2, 0, 6, 2], dtype=np.uint8))
                torch.manual_seed(0)
                dataset = EvalLoader(
                    filename_pattern="data/tiny_valid_*.bin",
                    seq_len=6,
                    process_rank=0,
                    num_processes=1,
                    tokenizer=TOKEN_IDS,
                )
                input_ids, labels, mask_rate = next(iter(dataset))
            finally:
                os.chdir(cwd)

        self.assertEqual(tuple(input_ids.shape), (6,))
        self.assertEqual(tuple(labels.shape), (6,))
        self.assertEqual(tuple(mask_rate.shape), (1,))
        self.assertTrue(torch.all(labels[(input_ids == TOKEN_IDS.cls_token_id)] == -100))

    def test_chunked_train_dataset_preserves_chunk_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            os.chdir(tmpdir)
            try:
                data_dir = Path("data")
                data_dir.mkdir()
                write_shard(
                    data_dir / "tiny_train_000000.bin",
                    np.array([0, 5, 2, 0, 6, 2, 0, 7, 2, 0, 8, 2], dtype=np.uint8),
                )
                dataset = ChunkedTrainDataset(
                    filename_pattern="data/tiny_train_*.bin",
                    max_length=4,
                    batch_size=2,
                    process_rank=0,
                    num_processes=1,
                    max_epochs=1,
                    tokenizer=TOKEN_IDS,
                    num_workers=1,
                )
                batch = next(iter(dataset))
            finally:
                os.chdir(cwd)

        self.assertEqual(tuple(batch.shape), (2, 4))
        self.assertEqual(batch.dtype, torch.int32)


if __name__ == "__main__":
    unittest.main()
