"""Dataset `_target_` for the packs written by src/data/prepack.py (memory-mapped)."""

from datasets import Dataset, load_from_disk


def load_prepacked(path: str) -> Dataset:
    """Load THD packs (input_ids, labels, position_ids, seq_lens, seq_lens_padded) from disk."""
    return load_from_disk(path)
