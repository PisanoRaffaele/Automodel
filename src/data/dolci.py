"""Dolci-Instruct-SFT parquet (minerva-post-training copy, with `isin_tulu3`) -> chat JSONL for prep.sources.

Same filters as LLaMA-Factory's pyutils/prep_dolci.py, plus the Tulu dedup:
  - drop rows whose prompt is in Tulu-3 (`isin_tulu3`), since Tulu is in the mix too;
  - drop tool use (minerva_chatml has no tool slot) and rows not ending in a non-empty assistant turn;
  - drop answers with thinking residue (<think> tags or >= 3 markers like "Wait,").
Line separators are escaped as in split.py.

Usage (login node):  ./run.sh src/data/dolci.py <config.yaml>   # reads prep.dolci.{parquet_dir,out}
"""

import collections
import glob
import json
import re
import sys

import pyarrow.parquet as pq
import yaml

ESCAPE = str.maketrans({c: f"\\u{ord(c):04x}" for c in "\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"})  # as in split.py
THINK_MARKERS = re.compile(
    r"\bWait,|\bHmm+[,.]|\bLet me (think|check|re-?examine|verify)|\bOkay, so\b|\bAlternatively, maybe\b|</?think>"
)


def reason(row: dict) -> str:
    """Why a row is dropped, or "kept"."""
    messages = row["messages"]
    if row["isin_tulu3"]:
        return "in_tulu3"
    if any(
        m["role"] not in ("system", "user", "assistant") or m.get("functions") or m.get("function_calls")
        for m in messages
    ):
        return "tool_use"
    if not messages or messages[-1]["role"] != "assistant" or not (messages[-1]["content"] or "").strip():
        return "no_target"
    for m in messages:
        c = m["content"] or ""
        if m["role"] == "assistant" and ("<think>" in c or "</think>" in c or len(THINK_MARKERS.findall(c)) >= 3):
            return "thinking_residue"
    return "kept"


dolci = yaml.safe_load(open(sys.argv[1]))["prep"]["dolci"]
stats = collections.Counter()
with open(dolci["out"], "w") as out:
    for f in sorted(glob.glob(f"{dolci['parquet_dir']}/*.parquet")):
        for batch in pq.ParquetFile(f).iter_batches(
            batch_size=2000, columns=["id", "messages", "source_dataset", "isin_tulu3"]
        ):
            for row in batch.to_pylist():
                why = reason(row)
                stats[why] += 1
                stats[f"{why}/{row['source_dataset']}"] += 1
                if why == "kept":
                    msgs = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
                    line = json.dumps(
                        {"id": row["id"], "source": row["source_dataset"], "messages": msgs}, ensure_ascii=False
                    )
                    out.write(line.translate(ESCAPE) + "\n")
for k, v in sorted(stats.items()):
    print(f"{k}: {v}")
