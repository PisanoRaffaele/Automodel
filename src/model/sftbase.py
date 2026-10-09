"""Build the SFT-ready base dir of a MinervaV2-20B-A3 pretraining HF checkpoint.

Mirrors what LLaMA-Factory did for the 4k runs (resize_vocab + desc_init + minerva_chatml):
  - 8 new special tokens (ChatML + reasoning/tool), ids 128000..128007, vocab padded to 128064
  - each new row (embed_tokens AND lm_head) = mean of the rows of its textual description
  - the 56 unused padding rows = mean row (never seen in data)
  - router_aux_loss_coef = 0, eos = <|im_end|>, generation_config with valid ids
  - ChatML template with {% generation %} markers -> AutoModel answer-only loss mask
Only the two shards holding embed_tokens / lm_head are rewritten; the others are symlinked.

Reads prep.base_ckpt / prep.special_tokens_desc, writes model.pretrained_model_name_or_path.

Usage (login node):  ./run.sh src/model/sftbase.py <config.yaml>
"""

import json
import os
import shutil
import sys

import yaml
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

cfg = yaml.safe_load(open(sys.argv[1]))
SRC, DESC = cfg["prep"]["base_ckpt"], cfg["prep"]["special_tokens_desc"]
DST = cfg["model"]["pretrained_model_name_or_path"]
PAD_TO = 64

# minerva_chatml (same rendering as the LF runs); assistant content + <|im_end|>\n is supervised.
CHAT_TEMPLATE = (
    "{{- '<|begin_of_text|>' }}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- generation %}{{- message['content'] + '<|im_end|>\\n' }}{%- endgeneration %}"
    "{%- else %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)

os.makedirs(DST, exist_ok=True)
desc = yaml.safe_load(open(DESC))
new_tokens = list(desc)

tok = AutoTokenizer.from_pretrained(SRC)
old_vocab = len(tok)
assert old_vocab == 128000, old_vocab
# description token ids computed with the ORIGINAL tokenizer
desc_ids = {t: tok(d, add_special_tokens=False).input_ids for t, d in desc.items()}
tok.add_special_tokens({"additional_special_tokens": new_tokens})
new_ids = tok.convert_tokens_to_ids(new_tokens)
assert new_ids == list(range(128000, 128000 + len(new_tokens))), new_ids
new_vocab = -(-len(tok) // PAD_TO) * PAD_TO
tok.eos_token = "<|im_end|>"
tok.chat_template = CHAT_TEMPLATE
tok.model_max_length = 131072
tok.save_pretrained(DST)
# save_pretrained writes chat_template.jinja; drop any stale copy-over from SRC
print(f"tokenizer: {old_vocab} -> {len(tok)} tokens, matrix rows -> {new_vocab}")

index = json.load(open(f"{SRC}/model.safetensors.index.json"))
wmap = index["weight_map"]
targets = {"model.embed_tokens.weight", "lm_head.weight"}
patched_shards = {wmap[k] for k in targets}

for shard in sorted(set(wmap.values())):
    dst = f"{DST}/{shard}"
    if os.path.lexists(dst):
        os.remove(dst)
    if shard not in patched_shards:
        os.symlink(f"{SRC}/{shard}", dst)
        continue
    tensors = load_file(f"{SRC}/{shard}")
    for key in targets & tensors.keys():
        w = tensors[key]
        assert w.shape[0] == old_vocab, (key, w.shape)
        w32 = w.float()
        out = w32.mean(0, keepdim=True).repeat(new_vocab, 1)
        out[:old_vocab] = w32
        for t, i in zip(new_tokens, new_ids):
            out[i] = w32[desc_ids[t]].mean(0)
        med = w32.norm(dim=1).median().item()
        newn = out[new_ids].norm(dim=1)
        print(f"{key}: {tuple(w.shape)} -> {tuple(out.shape)} | median norm {med:.4f} | new {newn.tolist()}")
        tensors[key] = out.to(w.dtype).contiguous()
    save_file(tensors, dst, metadata={"format": "pt"})
    print(f"wrote {dst}")

json.dump(index, open(f"{DST}/model.safetensors.index.json", "w"), indent=2)

cfg = json.load(open(f"{SRC}/config.json"))
cfg.update(
    vocab_size=new_vocab,
    router_aux_loss_coef=0.0,
    bos_token_id=tok.bos_token_id,
    eos_token_id=tok.eos_token_id,
    pad_token_id=tok.pad_token_id,
)
json.dump(cfg, open(f"{DST}/config.json", "w"), indent=2)
json.dump(
    {
        "bos_token_id": tok.bos_token_id,
        "eos_token_id": [tok.eos_token_id, tok.convert_tokens_to_ids("<|end_of_text|>")],
        "pad_token_id": tok.pad_token_id,
        "do_sample": True,
    },
    open(f"{DST}/generation_config.json", "w"),
    indent=2,
)
shutil.copy(DESC, f"{DST}/special_tokens_desc.yaml")

# sanity: template render + assistant mask
conv = [{"role": "user", "content": "Ciao!"}, {"role": "assistant", "content": "Ciao, come posso aiutarti?"}]
out = AutoTokenizer.from_pretrained(DST).apply_chat_template(
    conv, tokenize=True, return_dict=True, return_assistant_tokens_mask=True
)
ids, mask = out["input_ids"], out["assistant_masks"]
print(repr(tok.decode(ids)))
print("supervised:", repr(tok.decode([i for i, m in zip(ids, mask) if m])))
print("done:", DST)
