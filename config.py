import tiktoken, tiktoken.load
from pathlib import Path
import torch
import json

def get_tokenizer(tokenizer_path):
    # whoever decided that start_header_id and end_header_id should be between token 3 and 4 is evil!
    special_tokens = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|reserved_special_token_4|>",
        "<|eot_id|>"
    ] + [
        f"<|reserved_special_token_{i+5}|>" for i in range(256)
    ]
    mergeable_ranks = tiktoken.load.load_tiktoken_bpe(tokenizer_path)
    tokenizer = tiktoken.Encoding(
        name = Path(tokenizer_path).name,
        pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        mergeable_ranks = mergeable_ranks,
        special_tokens = {
            token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)
        }
    )
    return tokenizer

class Config:
    def __init__(self, path, device, dtype=torch.bfloat16):
        self.tokenizer = get_tokenizer(f"{path}/tokenizer.model")
        self.model  = torch.load(f"{path}/consolidated.00.pth", map_location=device)
        config = json.load(open(f"{path}/params.json"))
        self.dim        = config["dim"]
        self.n_layers   = config["n_layers"]
        self.n_heads    = config["n_heads"]
        self.n_kv_heads = config["n_kv_heads"]
        self.kv_mult    = self.n_heads // self.n_kv_heads
        self.head_dim   = self.dim // self.n_heads
        self.rot_dim    = self.head_dim // 2 * 2          # even part
        self.vocab_size = config["vocab_size"]
        self.norm_eps   = config["norm_eps"]
        self.rope_theta = torch.tensor(config["rope_theta"], device=device, dtype=dtype)