from pathlib import Path
import torch
import torch.nn.functional as F
import json
import time
import math
from tokenizer import get_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
with torch.no_grad():
    print("Loading model and tokenizer...")
    load_start = time.time()

    PATH = "/home/naklecha/.llama/checkpoints/Llama3.2-1B-Instruct"
    tokenizer = get_tokenizer(f"{PATH}/tokenizer.model")
    model = torch.load(f"{PATH}/consolidated.00.pth", map_location=device)
    with open(f"{PATH}/params.json", "r") as f: config = json.load(f)
    print(f"Loading took {time.time() - load_start:.3f}s")

    dim = config["dim"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    vocab_size = config["vocab_size"]
    norm_eps = config["norm_eps"]
    rope_theta = torch.tensor(config["rope_theta"])
    head_dim = dim // n_heads
    kv_multiple = n_heads // n_kv_heads

    prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nWhat is the meaning of life?Generate 1000 different possible answers.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    tokens = torch.tensor(tokenizer.encode(prompt, allowed_special="all"))
    prompt_split_as_tokens = [tokenizer.decode([token.item()]) for token in tokens]
    MAX_TOKENS_TO_GENERATE = 1000

    embedding_layer = torch.nn.Embedding(vocab_size, dim)
    embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])

    zero_to_one_split_into_64_parts = torch.tensor(range(32))/32
    freqs = 1.0 / (rope_theta ** zero_to_one_split_into_64_parts)
    freqs_for_each_token = torch.outer(torch.arange(2000), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)
    rms_norm = lambda tensor, norm_weights: (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) * norm_weights

    rot_dim = head_dim // 2 * 2
    k_cache, v_cache = [None]*n_layers, [None]*n_layers
    position = 0

    generation_start = time.time()
    prefill_start = time.time()
    for tok_id in tokens.tolist():
        x = embedding_layer(torch.tensor([tok_id], device=device)).to(torch.bfloat16)
        for layer in range(n_layers):
            x_norm = rms_norm(x, model[f"layers.{layer}.attention_norm.weight"])
            q_layer = model[f"layers.{layer}.attention.wq.weight"]
            k_layer = model[f"layers.{layer}.attention.wk.weight"]
            v_layer = model[f"layers.{layer}.attention.wv.weight"]
            q = torch.matmul(x_norm, q_layer.T).view(1, n_heads, head_dim)
            k = torch.matmul(x_norm, k_layer.T).view(1, n_kv_heads, head_dim)
            v = torch.matmul(x_norm, v_layer.T).view(1, n_kv_heads, head_dim)
            q_rotated, q_pass = q[..., :rot_dim], q[..., rot_dim:]
            k_rotated, k_pass = k[..., :rot_dim], k[..., rot_dim:]
            q_rotated = q_rotated.float().view(1, n_heads, rot_dim//2, 2)
            k_rotated = k_rotated.float().view(1, n_kv_heads, rot_dim//2, 2)
            freqs_layer = freqs_cis[position:position+1, :rot_dim//2].unsqueeze(1)
            q_rotated = torch.view_as_real(torch.view_as_complex(q_rotated) * freqs_layer)
            k_rotated = torch.view_as_real(torch.view_as_complex(k_rotated) * freqs_layer)
            q = torch.cat([q_rotated.view(1, n_heads, rot_dim), q_pass], dim=-1).to(torch.bfloat16)
            k = torch.cat([k_rotated.view(1, n_kv_heads, rot_dim), k_pass], dim=-1).to(torch.bfloat16)
            if kv_multiple > 1:
                k = k.repeat_interleave(kv_multiple, dim=1)
                v = v.repeat_interleave(kv_multiple, dim=1)
            if k_cache[layer] is None:
                k_cache[layer], v_cache[layer] = k, v
            else:
                k_cache[layer] = torch.cat([k_cache[layer], k], dim=0)
                v_cache[layer] = torch.cat([v_cache[layer], v], dim=0)
            k_all = k_cache[layer].permute(1, 0, 2)
            v_all = v_cache[layer].permute(1, 0, 2)
            q_ = q.view(1, n_heads, 1, head_dim).float()
            k_ = k_all.unsqueeze(0).float()
            v_ = v_all.unsqueeze(0).float()
            scores = torch.matmul(q_, k_.transpose(-2, -1)) / math.sqrt(head_dim)
            attention = torch.softmax(scores, dim=-1)
            attention_output = torch.matmul(attention, v_)
            attention_output = attention_output.to(torch.bfloat16).squeeze(2).reshape(1, dim)
            w_layer = model[f"layers.{layer}.attention.wo.weight"]
            x = x + torch.matmul(attention_output, w_layer.T)
            x_ffn_norm = rms_norm(x, model[f"layers.{layer}.ffn_norm.weight"])
            w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
            w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
            w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
            ff = torch.matmul(F.silu(torch.matmul(x_ffn_norm, w1.T)) * torch.matmul(x_ffn_norm, w3.T), w2.T)
            x = x + ff
        x = rms_norm(x, model["norm.weight"])
        logits = torch.matmul(x, model["output.weight"].T)
        last_logits = logits
        position += 1
    prefill_time = time.time() - prefill_start

    next_token = torch.argmax(last_logits, dim=-1).unsqueeze(0)
    output_tokens = []

    for i in range(MAX_TOKENS_TO_GENERATE):
        token_start = time.time()
        output_tokens.append(next_token.item())
        next_str = tokenizer.decode([output_tokens[-1]])
        print(f"[TOKEN {i+1}] {repr(next_str)} -- {time.time() - token_start:.3f}s")
        if next_str == "<|eot_id|>": break
        x = embedding_layer(next_token).to(torch.bfloat16)
        for layer in range(n_layers):
            x_norm = rms_norm(x, model[f"layers.{layer}.attention_norm.weight"])
            q_layer = model[f"layers.{layer}.attention.wq.weight"]
            k_layer = model[f"layers.{layer}.attention.wk.weight"]
            v_layer = model[f"layers.{layer}.attention.wv.weight"]
            q = torch.matmul(x_norm, q_layer.T).view(1, n_heads, head_dim)
            k = torch.matmul(x_norm, k_layer.T).view(1, n_kv_heads, head_dim)
            v = torch.matmul(x_norm, v_layer.T).view(1, n_kv_heads, head_dim)
            q_rotated, q_pass = q[..., :rot_dim], q[..., rot_dim:]
            k_rotated, k_pass = k[..., :rot_dim], k[..., rot_dim:]
            q_rotated = q_rotated.float().view(1, n_heads, rot_dim//2, 2)
            k_rotated = k_rotated.float().view(1, n_kv_heads, rot_dim//2, 2)
            freqs_layer = freqs_cis[position:position+1, :rot_dim//2].unsqueeze(1)
            q_rotated = torch.view_as_real(torch.view_as_complex(q_rotated) * freqs_layer)
            k_rotated = torch.view_as_real(torch.view_as_complex(k_rotated) * freqs_layer)
            q = torch.cat([q_rotated.view(1, n_heads, rot_dim), q_pass], dim=-1).to(torch.bfloat16)
            k = torch.cat([k_rotated.view(1, n_kv_heads, rot_dim), k_pass], dim=-1).to(torch.bfloat16)
            if kv_multiple > 1:
                k = k.repeat_interleave(kv_multiple, dim=1)
                v = v.repeat_interleave(kv_multiple, dim=1)
            k_cache[layer] = torch.cat([k_cache[layer], k], dim=0)
            v_cache[layer] = torch.cat([v_cache[layer], v], dim=0)
            k_all = k_cache[layer].permute(1, 0, 2)
            v_all = v_cache[layer].permute(1, 0, 2)
            q_ = q.view(1, n_heads, 1, head_dim).float()
            k_ = k_all.unsqueeze(0).float()
            v_ = v_all.unsqueeze(0).float()
            scores = torch.matmul(q_, k_.transpose(-2, -1)) / math.sqrt(head_dim)
            attention = torch.softmax(scores, dim=-1)
            attention_output = torch.matmul(attention, v_)
            attention_output = attention_output.to(torch.bfloat16).squeeze(2).reshape(1, dim)
            w_layer = model[f"layers.{layer}.attention.wo.weight"]
            x = x + torch.matmul(attention_output, w_layer.T)
            x_ffn_norm = rms_norm(x, model[f"layers.{layer}.ffn_norm.weight"])
            w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
            w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
            w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
            ff = torch.matmul(F.silu(torch.matmul(x_ffn_norm, w1.T)) * torch.matmul(x_ffn_norm, w3.T), w2.T)
            x = x + ff
        x = rms_norm(x, model["norm.weight"])
        logits = torch.matmul(x, model["output.weight"].T)
        next_token = torch.argmax(logits, dim=-1)
        position += 1

    time_taken = time.time() - generation_start
    print("PREFILL TIME", prefill_time)
    print("TOTAL TOKENS GENERATED", len(output_tokens))
    print("GENERATION TIME", time_taken)
    print("OUTPUT TOKENS PER SECOND", len(output_tokens) / time_taken)
    print("[END] Generation complete")
    print("[TOKENS]", output_tokens)