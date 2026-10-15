"""Make prep.sources and a reference SFT set (prep.dedup.ref) disjoint, for mixing them in one run.

- prep.dedup.out_dir/<source>-dedup.train.jsonl: the train rows of each prep.sources file (the split.py split) minus
  rows of ref (same original id within the same language, or same first prompt) and minus
  prep.dedup.drop_sources (e.g. hard-coded identity data).
- prep.dedup.out_dir/<ref name>.train.jsonl: ref minus the sources' val rows, so the shared
  validation packs stay unseen; `conversations` written as `messages`.
The tulu / dromedario JSONLs carry no id: prep.dedup.ids gives, per source, the HF arrow shards in
the same row order, their messages column and the ref macrodataset its ids are compared with.
Line separators are escaped as in split.py.

Usage (login node):  ./run.sh src/data/dedup.py <config.yaml>
"""

import collections
import glob
import hashlib
import json
import os
import sys

import pyarrow as pa
import yaml

ESCAPE = str.maketrans({c: f"\\u{ord(c):04x}" for c in "\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"})  # as in split.py

prep = yaml.safe_load(open(sys.argv[1]))["prep"]
dedup = prep["dedup"]


def prompt_key(messages: list) -> bytes:
    """Digest of the first non-system turn, case and whitespace normalized."""
    text = next((m["content"] or "" for m in messages if m["role"] != "system"), "")
    return hashlib.md5(" ".join(text.split()).lower().encode()).digest()


def ref_rows():
    """(row, original id) for each ref row."""
    with open(dedup["ref"]) as f:
        for line in f:
            row = json.loads(line)
            yield row, row["id"].split("|", 2)[-1]  # "<macrodataset>|<microdataset>|<original id>"


def arrow_rows(spec: dict):
    """(id, source, first content) of the HF rows behind a JSONL without ids, rows lacking messages skipped."""
    for path in sorted(glob.glob(spec["arrow"])):
        with pa.memory_map(path) as f:
            for batch in pa.ipc.open_stream(f):
                for row in batch.select(["id", "source", spec["messages"]]).to_pylist():
                    if row[spec["messages"]] and len(row[spec["messages"]]) >= 2:
                        yield row["id"], row["source"], row[spec["messages"]][0]["content"]


ref_ids, ref_keys = collections.defaultdict(set), set()
for row, oid in ref_rows():
    ref_ids[row["macrodataset"]].add(oid)
    ref_keys.add(prompt_key(row["conversations"]))

val_ids, val_keys = collections.defaultdict(set), set()
for src in prep["sources"]:
    stem = src.removesuffix(".jsonl")
    name = os.path.basename(stem)
    spec = dedup["ids"].get(name)
    hf = arrow_rows(spec) if spec else None
    stats = collections.Counter()
    with open(src) as f, open(f"{dedup['out_dir']}/{name}-dedup.train.jsonl", "w") as out:
        for n, line in enumerate(f, start=1):
            row = json.loads(line)
            if hf:
                oid, source, first = next(hf)
                assert first == row["messages"][0]["content"], f"{name} row {n}: JSONL and arrow out of order"
            else:
                oid, source = row["id"], row["source"]
            key = prompt_key(row["messages"])
            same_lang_ids = ref_ids[spec["ref"]] if spec else set()
            if n % prep["val_every"] == 0:
                val_keys.add(key)
                if spec:
                    val_ids[spec["ref"]].add(oid)
                why = "val"
            elif source in dedup["drop_sources"]:
                why = "drop_source"
            elif oid in same_lang_ids:
                why = "in_ref_id"
            elif key in ref_keys:
                why = "in_ref_prompt"
            else:
                why = "kept"
                out.write(line.translate(ESCAPE))
            stats[why] += 1
    if hf:
        assert next(hf, None) is None, f"{name}: arrow has rows the JSONL lacks"
    print(f"{name}: {dict(stats)}", flush=True)

ref_out = os.path.join(dedup["out_dir"], os.path.basename(dedup["ref"]).removesuffix(".jsonl"))
stats = collections.Counter()
with open(f"{ref_out}.train.jsonl", "w") as out:
    for row, oid in ref_rows():
        if oid in val_ids[row["macrodataset"]] or prompt_key(row["conversations"]) in val_keys:
            stats["in_val"] += 1
            continue
        stats["kept"] += 1
        msgs = [{"role": m["role"], "content": m["content"]} for m in row["conversations"]]
        out.write(json.dumps({"id": row["id"], "messages": msgs}, ensure_ascii=False).translate(ESCAPE) + "\n")
print(f"{os.path.basename(ref_out)}: {dict(stats)}")
