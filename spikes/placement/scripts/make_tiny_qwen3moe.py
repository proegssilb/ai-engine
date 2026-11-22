#!/usr/bin/env python3
"""Write a small random-weight qwen3moe GGUF for placement experiments.

The weights are noise: the model produces garbage text. That's fine; the
spike checks where tensors live and whether outputs are identical across
placements, not whether the model is any good.

Usage: python3 make_tiny_qwen3moe.py OUT.gguf [--experts N] [--layers N]
Needs numpy and the gguf-py package from vendor/llama.cpp/gguf-py.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "vendor/llama.cpp/gguf-py"))
import gguf  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--layers", type=int, default=4)
p.add_argument("--embd", type=int, default=512)
p.add_argument("--experts", type=int, default=32)
p.add_argument("--experts-used", type=int, default=4)
p.add_argument("--ff-exp", type=int, default=512)
p.add_argument("--heads", type=int, default=8)
p.add_argument("--kv-heads", type=int, default=4)
p.add_argument("--seed", type=int, default=1234)
a = p.parse_args()

head_dim = a.embd // a.heads
rng = np.random.default_rng(a.seed)

# Byte-fallback SentencePiece vocab: 3 specials + 256 byte tokens.
tokens = [b"<unk>", b"<s>", b"</s>"] + [f"<0x{i:02X}>".encode() for i in range(256)]
types = [gguf.TokenType.UNKNOWN, gguf.TokenType.CONTROL, gguf.TokenType.CONTROL] + [gguf.TokenType.BYTE] * 256
n_vocab = len(tokens)


def w(*shape, dtype=np.float16):
    # numpy shape is reversed relative to ggml ne[]
    return (rng.standard_normal(shape, dtype=np.float32) * 0.02).astype(dtype)


def ones(n):
    return np.ones(n, dtype=np.float32)


g = gguf.GGUFWriter(a.out, "qwen3moe")
g.add_name("tiny-random-qwen3moe")
g.add_context_length(4096)
g.add_embedding_length(a.embd)
g.add_block_count(a.layers)
g.add_feed_forward_length(a.ff_exp * a.experts_used)
g.add_expert_feed_forward_length(a.ff_exp)
g.add_head_count(a.heads)
g.add_head_count_kv(a.kv_heads)
g.add_key_length(head_dim)
g.add_value_length(head_dim)
g.add_expert_count(a.experts)
g.add_expert_used_count(a.experts_used)
g.add_layer_norm_rms_eps(1e-6)
g.add_rope_freq_base(1_000_000.0)
g.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

g.add_tokenizer_model("llama")
g.add_token_list(tokens)
g.add_token_types(types)
g.add_token_scores([0.0] * n_vocab)
g.add_unk_token_id(0)
g.add_bos_token_id(1)
g.add_eos_token_id(2)

g.add_tensor("token_embd.weight", w(n_vocab, a.embd))
g.add_tensor("output_norm.weight", ones(a.embd))
g.add_tensor("output.weight", w(n_vocab, a.embd))
for i in range(a.layers):
    b = f"blk.{i}"
    g.add_tensor(f"{b}.attn_norm.weight", ones(a.embd))
    g.add_tensor(f"{b}.attn_q.weight", w(a.heads * head_dim, a.embd))
    g.add_tensor(f"{b}.attn_k.weight", w(a.kv_heads * head_dim, a.embd))
    g.add_tensor(f"{b}.attn_v.weight", w(a.kv_heads * head_dim, a.embd))
    g.add_tensor(f"{b}.attn_output.weight", w(a.embd, a.heads * head_dim))
    g.add_tensor(f"{b}.attn_q_norm.weight", ones(head_dim))
    g.add_tensor(f"{b}.attn_k_norm.weight", ones(head_dim))
    g.add_tensor(f"{b}.ffn_norm.weight", ones(a.embd))
    g.add_tensor(f"{b}.ffn_gate_inp.weight", w(a.experts, a.embd, dtype=np.float32))
    g.add_tensor(f"{b}.ffn_gate_exps.weight", w(a.experts, a.ff_exp, a.embd))
    g.add_tensor(f"{b}.ffn_up_exps.weight", w(a.experts, a.ff_exp, a.embd))
    g.add_tensor(f"{b}.ffn_down_exps.weight", w(a.experts, a.embd, a.ff_exp))

g.write_header_to_file()
g.write_kv_data_to_file()
g.write_tensors_to_file()
g.close()
print(f"wrote {a.out}")
