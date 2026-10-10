"""Tokenize + bin-pack the SFT splits once, offline, into memory-mapped THD packs.

Loader-side packing makes every rank tokenize all 1.55M rows and keep them in RAM (host OOM at
4 ranks/node), and AutoModel's sequential packer fills only ~86% of each pack (~1171 steps vs
LLaMA-Factory's 1008). Here: ChatDataset (same tokenizer/template/truncation as training) ->
greedy knapsack (LF's neat_packing algorithm) -> AutoModel's own `_tensorize_and_pad_pack`, so the
THD collator sees the usual layout. Writes dataset.path and validation_dataset.path.

Usage (CPU job):  ./run.sh slurm/prepack.sbatch.sh <config.yaml>
"""

import multiprocessing as mp
import os
import shutil
import sys

import numpy as np
import yaml
from datasets import Dataset, Features, Sequence, Value
from transformers import AutoTokenizer

from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset
from nemo_automodel.components.datasets.llm.neat_packing import greedy_knapsack
from nemo_automodel.components.datasets.llm.packed_sequence import (
    CROSS_ENTROPY_IGNORE_IDX,
    _fill_labels_with_cross_entropy_ignore_idx,
    _tensorize_and_pad_pack,
)

CHUNK = 5000
cfg = yaml.safe_load(open(sys.argv[1]))
PACK = cfg["packed_sequence"]["packed_sequence_size"]
tok = AutoTokenizer.from_pretrained(cfg["model"]["pretrained_model_name_or_path"])
ds, TMP = None, None  # set per split; pool workers inherit them via fork


def tokenize(start: int) -> str:
    """Rows [start, start + CHUNK) -> one int32 shard on disk (kept, so a rerun resumes)."""
    path = f"{TMP}/{start:09d}.npz"
    if not os.path.exists(path):
        ids, labels, lens = [], [], []
        for i in range(start, min(start + CHUNK, len(ds))):
            s = ds[i]
            lab = s["labels"]
            if s.get("loss_mask"):  # same handling as packed_sequence.pack_dataset
                lab = _fill_labels_with_cross_entropy_ignore_idx(lab, s["loss_mask"])
            ids += s["input_ids"]
            labels += lab
            lens.append(len(s["input_ids"]))
        np.savez(path + ".part.npz", ids=np.array(ids, np.int32), labels=np.array(labels, np.int32), lens=lens)
        os.replace(path + ".part.npz", path)
    return path


def packs():
    """Yield one THD pack per knapsack bin (reads the split's bins/ids/labels/offsets globals)."""
    for b in bins:
        pack = {"input_ids": [], "labels": [], "position_ids": [], "seq_lens": []}
        for i in b:
            lo, hi = offsets[i], offsets[i + 1]
            pack["input_ids"] += ids[lo:hi].tolist()
            pack["labels"] += labels[lo:hi].tolist()
            pack["position_ids"] += range(hi - lo)
            pack["seq_lens"].append(int(hi - lo))
        out = _tensorize_and_pad_pack(pack, tok.pad_token_id, PACK, CROSS_ENTROPY_IGNORE_IDX)
        yield {k: v.tolist() for k, v in out.items()}


if __name__ == "__main__":
    for split, key in (("val", "validation_dataset"), ("train", "dataset")):
        out = cfg[key]["path"]
        files = [src.removesuffix(".jsonl") + f".{split}.jsonl" for src in cfg["prep"]["sources"]]
        ds, TMP = ChatDataset(files, tok, seq_length=PACK, truncation=True), out + ".tmp"
        os.makedirs(TMP, exist_ok=True)
        with mp.get_context("fork").Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 4))) as pool:
            shards = pool.map(tokenize, range(0, len(ds), CHUNK))
        ds = None  # the raw rows are no longer needed; free them before packing

        # One flat token stream per field, filled shard by shard (no list of arrays + concatenate copy).
        lens = np.concatenate([np.load(f)["lens"] for f in shards]).astype(np.int64)
        offsets = np.concatenate([[0], np.cumsum(lens)])
        ids, labels = np.empty(offsets[-1], np.int32), np.empty(offsets[-1], np.int32)
        pos = 0
        for f in shards:
            shard = np.load(f)
            n = shard["ids"].size
            ids[pos : pos + n], labels[pos : pos + n] = shard["ids"], shard["labels"]
            pos += n

        bins = greedy_knapsack(lens.tolist(), PACK)
        print(
            f"{split}: {len(lens)} rows, {lens.sum()} tok -> {len(bins)} packs, fill {lens.sum() / len(bins) / PACK:.2%}"
            f", {len(bins) / cfg['step_scheduler']['global_batch_size']:.1f} steps",
            flush=True,
        )
        features = Features(
            {
                k: Sequence(Value("int32"))
                for k in ("input_ids", "labels", "position_ids", "seq_lens", "seq_lens_padded")
            }
        )
        Dataset.from_generator(packs, features=features, cache_dir=out + ".cache").save_to_disk(out)
        shutil.rmtree(TMP), shutil.rmtree(out + ".cache")
        print(f"saved {out}", flush=True)
