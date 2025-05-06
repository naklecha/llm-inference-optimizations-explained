from pathlib import Path
import torch
import json
import time
from prompts import prompts
from tokenizer import get_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
with torch.no_grad():
    batch_size = len(prompts)
    print(f"[BATCH] {batch_size} prompt(s) will be processed together.")

    print("Loading model and tokenizer...")
    load_start = time.time()

    PATH = "/home/naklecha/.llama/checkpoints/Llama3.2-1B-Instruct"
    tokenizer = get_tokenizer(f"{PATH}/tokenizer.model")
    model = torch.load(f"{PATH}/consolidated.00.pth", map_location=device)
    with open(f"{PATH}/params.json", "r") as f: config = json.load(f)

    dim = config["dim"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    vocab_size = config["vocab_size"]
    norm_eps = config["norm_eps"]
    rope_theta = torch.tensor(config["rope_theta"])
    head_dim = dim // n_heads
    kv_multiple = n_heads // n_kv_heads

    print(f"Loading took {time.time() - load_start:.3f}s")

    embedding_layer = torch.nn.Embedding(vocab_size, dim)
    embedding_layer.weight.data.copy_(model["tok_embeddings.weight"].to(torch.bfloat16))

    zero_to_one_split_into_64_parts = torch.tensor(range(32))/32
    freqs = 1.0 / (rope_theta ** zero_to_one_split_into_64_parts)
    freqs_for_each_token = torch.outer(torch.arange(2048), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)
    rms_norm = lambda tensor, norm_weights: (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) * norm_weights
    rot_dim = head_dim // 2 * 2

    prompt_tokens = []
    for p in prompts:
        prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n" + p + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        prompt_tokens.append(tokenizer.encode(prompt, allowed_special="all"))

    max_prompt_len = max(len(toks) for toks in prompt_tokens)
    print(f"[PREFILL] Longest prompt length: {max_prompt_len} tokens")

    pad_id = tokenizer.encode("<|end_of_text|>", allowed_special="all")[0]
    eot_id = tokenizer.encode("<|eot_id|>", allowed_special="all")[0]

    for i in range(batch_size):
        pad_len = max_prompt_len - len(prompt_tokens[i])
        if pad_len:
            prompt_tokens[i].extend([pad_id]*pad_len)

    prompt_tokens = torch.tensor(prompt_tokens)

    k_cache = [torch.empty((0, batch_size, n_heads, head_dim), dtype=torch.bfloat16, device=device) for _ in range(n_layers)]
    v_cache = [torch.empty((0, batch_size, n_heads, head_dim), dtype=torch.bfloat16, device=device) for _ in range(n_layers)]

    for position in range(max_prompt_len):
        tok_step = prompt_tokens[:, position]
        x = embedding_layer(tok_step).to(torch.bfloat16)
        for layer in range(n_layers):
            x_norm = rms_norm(x, model[f"layers.{layer}.attention_norm.weight"])
            q = torch.matmul(x_norm, model[f"layers.{layer}.attention.wq.weight"].T)
            q = q.view(batch_size, n_heads, head_dim)
            q_rotated, q_pass = q[..., :rot_dim], q[..., rot_dim:]
            q_rotated = q_rotated.float().view(batch_size, n_heads, rot_dim//2, 2)
            freqs_layer = freqs_cis[position:position+1, :rot_dim//2]
            q_rotated = torch.view_as_real(torch.view_as_complex(q_rotated) * freqs_layer)
            q = torch.cat([q_rotated.view(batch_size, n_heads, rot_dim), q_pass], dim=-1).to(torch.bfloat16)
            k = torch.matmul(x_norm, model[f"layers.{layer}.attention.wk.weight"].T).view(batch_size, n_kv_heads, head_dim)
            v = torch.matmul(x_norm, model[f"layers.{layer}.attention.wv.weight"].T).view(batch_size, n_kv_heads, head_dim)
            k_rotated, k_pass = k[..., :rot_dim], k[..., rot_dim:]
            k_rotated = k_rotated.float().view(batch_size, n_kv_heads, rot_dim//2, 2)
            k_rotated = torch.view_as_real(torch.view_as_complex(k_rotated) * freqs_layer)
            k = torch.cat([k_rotated.view(batch_size, n_kv_heads, rot_dim), k_pass], dim=-1).to(torch.bfloat16)
            if kv_multiple > 1:
                k = k.repeat_interleave(kv_multiple, dim=1)
                v = v.repeat_interleave(kv_multiple, dim=1)
            k_cache[layer] = torch.cat([k_cache[layer], k.unsqueeze(0)], dim=0)
            v_cache[layer] = torch.cat([v_cache[layer], v.unsqueeze(0)], dim=0)
            q = q.unsqueeze(2)
            k_all = k_cache[layer].permute(1, 2, 0, 3)
            v_all = v_cache[layer].permute(1, 2, 0, 3)
            scores = torch.matmul(q, k_all.transpose(-2, -1)) / (head_dim ** 0.5)
            attention = torch.softmax(scores, dim=-1).to(torch.bfloat16)
            attention_output = torch.matmul(attention, v_all)
            attention_output = attention_output.squeeze(2).reshape(batch_size, dim)
            x = x + torch.matmul(attention_output, model[f"layers.{layer}.attention.wo.weight"].T)
            x_ffn_norm = rms_norm(x, model[f"layers.{layer}.ffn_norm.weight"])
            w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
            w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
            w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
            ff = torch.matmul(torch.nn.functional.silu(torch.matmul(x_ffn_norm, w1.T)) * torch.matmul(x_ffn_norm, w3.T), w2.T)
            x = x + ff
        x = rms_norm(x, model["norm.weight"])
        logits = torch.matmul(x, model["output.weight"].T)
        last_logits = logits

    MAX_TOKENS_TO_GENERATE = 1000
    next_token = torch.argmax(last_logits, dim=-1)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    output_tokens = [[] for _ in range(batch_size)]
    generation_start = time.time()
    step = 0

    while step < MAX_TOKENS_TO_GENERATE and not torch.all(finished):
        decoded = [tokenizer.decode([token.item()]) for token in next_token]
        for b, s in enumerate(decoded):
            if not finished[b]:
                output_tokens[b].append(s)
                if next_token[b].item() == eot_id:
                    finished[b] = True
                    print(f"Finished generation for prompt {b}")
        if torch.all(finished):
            break
        x = embedding_layer(next_token).to(torch.bfloat16)
        pos = k_cache[0].shape[0]
        for layer in range(n_layers):
            x_norm = rms_norm(x, model[f"layers.{layer}.attention_norm.weight"])
            q = torch.matmul(x_norm, model[f"layers.{layer}.attention.wq.weight"].T).view(batch_size, n_heads, head_dim)
            q_rotated, q_pass = q[..., :rot_dim], q[..., rot_dim:]
            q_rotated = q_rotated.float().view(batch_size, n_heads, rot_dim//2, 2)
            freqs_layer = freqs_cis[pos:pos+1, :rot_dim//2]
            q_rotated = torch.view_as_real(torch.view_as_complex(q_rotated) * freqs_layer)
            q = torch.cat([q_rotated.view(batch_size, n_heads, rot_dim), q_pass], dim=-1).to(torch.bfloat16)
            k = torch.matmul(x_norm, model[f"layers.{layer}.attention.wk.weight"].T).view(batch_size, n_kv_heads, head_dim)
            v = torch.matmul(x_norm, model[f"layers.{layer}.attention.wv.weight"].T).view(batch_size, n_kv_heads, head_dim)
            k_rotated, k_pass = k[..., :rot_dim], k[..., rot_dim:]
            k_rotated = k_rotated.float().view(batch_size, n_kv_heads, rot_dim//2, 2)
            k_rotated = torch.view_as_real(torch.view_as_complex(k_rotated) * freqs_layer)
            k = torch.cat([k_rotated.view(batch_size, n_kv_heads, rot_dim), k_pass], dim=-1).to(torch.bfloat16)
            if kv_multiple > 1:
                k = k.repeat_interleave(kv_multiple, dim=1)
                v = v.repeat_interleave(kv_multiple, dim=1)
            k_cache[layer] = torch.cat([k_cache[layer], k.unsqueeze(0)], dim=0)
            v_cache[layer] = torch.cat([v_cache[layer], v.unsqueeze(0)], dim=0)
            q = q.unsqueeze(2)
            k_all = k_cache[layer].permute(1, 2, 0, 3)
            v_all = v_cache[layer].permute(1, 2, 0, 3)
            scores = torch.matmul(q, k_all.transpose(-2, -1)) / (head_dim ** 0.5)
            attention = torch.softmax(scores, dim=-1).to(torch.bfloat16)
            attention_output = torch.matmul(attention, v_all)
            attention_output = attention_output.squeeze(2).reshape(batch_size, dim)
            x = x + torch.matmul(attention_output, model[f"layers.{layer}.attention.wo.weight"].T)
            x_ffn_norm = rms_norm(x, model[f"layers.{layer}.ffn_norm.weight"])
            w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
            w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
            w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
            ff = torch.matmul(torch.nn.functional.silu(torch.matmul(x_ffn_norm, w1.T)) * torch.matmul(x_ffn_norm, w3.T), w2.T)
            x = x + ff
        x = rms_norm(x, model["norm.weight"])
        logits = torch.matmul(x, model["output.weight"].T)
        next_token = torch.argmax(logits, dim=-1)
        step += 1

    time_taken = time.time() - generation_start
    total_tokens = sum(len(out) for out in output_tokens)
    print("[GENERATION TIME] ", time_taken)
    print("[TOTAL TOKENS GENERATED] ", total_tokens)
    print("[OUTPUT TOKENS PER SECOND] ", total_tokens / time_taken)
    print("[END] Generation complete")