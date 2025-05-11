import time
import torch
import torch.nn.functional as F
from prompts import prompts
from config import Config

def get_freqs_cis(max_tokens: int, *, rope_theta: float, device: torch.device):
    idx = torch.arange(32, device=device) / 32.0
    freqs_for_each_token = torch.outer(torch.arange(max_tokens, device=device), 1.0 / (rope_theta ** idx))
    return torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float):
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * weight

def apply_rope(t, freqs_layer, batch_size, rot_dim):
    t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t_rot = t_rot.float().view(batch_size, -1, rot_dim // 2, 2)
    t_rot = torch.view_as_real(torch.view_as_complex(t_rot) * freqs_layer)
    return torch.cat((t_rot.view(batch_size, -1, rot_dim), t_pass), dim=-1).to(torch.bfloat16)

def forward_transformer_block(x, model, k_cache, v_cache, pos, config, freqs_cis):
    for layer_index in range(config.n_layers):
        w_rms = model[f"layers.{layer_index}.attention_norm.weight"]
        x_norm = rms_norm(x, w_rms, config.norm_eps)
        q = (x_norm @ model[f"layers.{layer_index}.attention.wq.weight"].T).view(config.batch_size, config.n_heads, config.head_dim)
        k = (x_norm @ model[f"layers.{layer_index}.attention.wk.weight"].T).view(config.batch_size, config.n_kv_heads, config.head_dim)
        v = (x_norm @ model[f"layers.{layer_index}.attention.wv.weight"].T).view(config.batch_size, config.n_kv_heads, config.head_dim)
        freqs_layer = freqs_cis[pos : pos + 1, : config.rot_dim // 2]
        q = apply_rope(q, freqs_layer, config.batch_size, config.rot_dim)
        k = apply_rope(k, freqs_layer, config.batch_size, config.rot_dim)
        if config.kv_mult > 1:
            k = k.repeat_interleave(config.kv_mult, dim=1)
            v = v.repeat_interleave(config.kv_mult, dim=1)
        k_cache[layer_index] = torch.cat((k_cache[layer_index], k.unsqueeze(0)), dim=0)
        v_cache[layer_index] = torch.cat((v_cache[layer_index], v.unsqueeze(0)), dim=0)
        k_all = k_cache[layer_index].permute(1, 2, 0, 3)
        v_all = v_cache[layer_index].permute(1, 2, 0, 3)
        attn_out = F.scaled_dot_product_attention(q.unsqueeze(2), k_all, v_all, scale=1.0 / (config.head_dim ** 0.5), is_causal=False).squeeze(2)
        attn_out = attn_out.reshape(config.batch_size, config.dim)
        x = x + attn_out @ model[f"layers.{layer_index}.attention.wo.weight"].T
        w_rms_ffn = model[f"layers.{layer_index}.ffn_norm.weight"]
        x_ffn_norm = rms_norm(x, w_rms_ffn, config.norm_eps)
        w1, w2, w3 = (model[f"layers.{layer_index}.feed_forward.{w}.weight"] for w in ("w1", "w2", "w3"))
        ffn_out = (torch.nn.functional.silu(x_ffn_norm @ w1.T) * (x_ffn_norm @ w3.T)) @ w2.T
        x = x + ffn_out
    return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

with torch.no_grad():
    config = Config(path="/home/naklecha/.llama/checkpoints/Llama3.2-1B-Instruct", device=device)
    tokenizer, model = config.tokenizer, config.model
    embed = torch.nn.Embedding(config.vocab_size, config.dim, _weight=model["tok_embeddings.weight"]).to(device)
    formatted = [
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n" + p + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        for p in prompts
    ]
    config.batch_size = len(formatted)
    enc = [torch.tensor(tokenizer.encode(p, allowed_special="all"), dtype=torch.long) for p in formatted]
    max_len = max(len(t) for t in enc)
    pad_id = tokenizer.encode("<|end_of_text|>", allowed_special="all")[0]
    eot_id = tokenizer.encode("<|eot_id|>", allowed_special="all")[0]
    
    MAX_TOKENS_TO_GENERATE = 500
    MAX_TOTAL_TOKENS = max_len + MAX_TOKENS_TO_GENERATE
    
    freqs_cis = get_freqs_cis(MAX_TOTAL_TOKENS, rope_theta=config.rope_theta, device=device)
    config.rot_dim = config.head_dim // 2 * 2

    prompt_tokens = torch.full((config.batch_size, max_len), pad_id, dtype=torch.long, device=device)
    for i, t in enumerate(enc): prompt_tokens[i, : t.numel()] = t.to(device)

    k_cache = [torch.empty((0, config.batch_size, config.n_heads, config.head_dim), dtype=torch.bfloat16, device=device) for _ in range(config.n_layers)]
    v_cache = [torch.empty((0, config.batch_size, config.n_heads, config.head_dim), dtype=torch.bfloat16, device=device) for _ in range(config.n_layers)]

    print(f"[PREFILL] Processing {max_len} prompt tokens …")
    gen_start = prefill_start = time.time()
    for pos in range(max_len):
        tok_step = prompt_tokens[:, pos]
        x = embed(tok_step).to(torch.bfloat16)
        x = forward_transformer_block(x, model, k_cache, v_cache, pos, config, freqs_cis)
        if pos == max_len - 1:
            x_last = rms_norm(x, model["norm.weight"], config.norm_eps)
            last_logits = (x_last @ model["output.weight"].T).float()
    prefill_end = time.time()

    next_token = torch.argmax(last_logits, dim=-1)
    finished = torch.zeros(config.batch_size, dtype=torch.bool, device=device)
    generated = [[] for _ in range(config.batch_size)]

    step = 0
    while step < MAX_TOKENS_TO_GENERATE and not torch.all(finished):
        token_start = time.time()
        changed = False
        for i, token in enumerate(next_token):
            if not finished[i]:
                generated[i].append(token.item())
                if token == eot_id:
                    finished[i] = changed = True
                
        if changed:
            active_mask = ~finished
            finished = finished[active_mask]
            next_token = next_token[active_mask]
            x = embed(next_token).to(torch.bfloat16).to(device)
            for layer_index in range(config.n_layers):
                k_cache[layer_index] = k_cache[layer_index][:, active_mask, :, :]
                v_cache[layer_index] = v_cache[layer_index][:, active_mask, :, :]
            config.batch_size = len(next_token)
        else:
            x = embed(next_token).to(torch.bfloat16).to(device)
            
        pos = k_cache[0].shape[0]
        x = forward_transformer_block(x, model, k_cache, v_cache, pos, config, freqs_cis)
        x = rms_norm(x, model["norm.weight"], config.norm_eps)
        logits = (x @ model["output.weight"].T).float()
        next_token = torch.argmax(logits, dim=-1)
        step += 1

        token_end = time.time()
        print(f"[STEP {step}] [TOKENS GENERATED IN THIS STEP {config.batch_size}] -- {token_end - token_start:.3f}sec -- {config.batch_size / (token_end - token_start):.1f}tps")

    gen_time = time.time() - gen_start
    
    for i in range(config.batch_size):
        full = enc[i].tolist() + generated[i]
        print(f"\n\nPROMPT {i+1}")
        print(tokenizer.decode(full))
        
    total_new = sum(len(lst) for lst in generated)
    print(f"\n\n[PREFILL] Done in {prefill_end - prefill_start:.2f}s")
    print(f"[GENERATION] Produced {total_new} tokens in {gen_time:.2f}s — {total_new / gen_time:.1f} tok/s")