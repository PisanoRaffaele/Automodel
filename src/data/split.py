"""Split each prep.sources JSONL into <name>.train.jsonl / <name>.val.jsonl (every prep.val_every-th row -> val).

ChatDataset reads JSONL with str.splitlines(), which also breaks on U+2028/U+2029/U+0085 and
\\x0b \\x0c \\x1c-\\x1e; those only occur inside JSON strings, so they are written as \\uXXXX escapes
(every row still parses to the same text).

Usage (login node):  ./run.sh src/data/split.py <config.yaml>
"""

import sys

import yaml

ESCAPE = str.maketrans({c: f"\\u{ord(c):04x}" for c in "\x0b\x0c\x1c\x1d\x1e\x85  "})

prep = yaml.safe_load(open(sys.argv[1]))["prep"]
for src in prep["sources"]:
    stem = src.removesuffix(".jsonl")
    with open(src) as f, open(f"{stem}.train.jsonl", "w") as train, open(f"{stem}.val.jsonl", "w") as val:
        for n, line in enumerate(f, start=1):
            (val if n % prep["val_every"] == 0 else train).write(line.translate(ESCAPE))
    print(f"{src}: {n - n // prep['val_every']} train / {n // prep['val_every']} val")
