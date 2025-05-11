from prompts import prompts
from config import Config
import torch
import time


def get_freqs_cis(max_tokens: int, *, rope_theta: float, device: torch.device):
    idx = torch.arange(32, device=device) / 32.0
    freqs = 1.0 / (rope_theta ** idx)
    freqs_for_each_token = torch.outer(torch.arange(max_tokens, device=device), freqs)
    return torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)

def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor):
    rot_dim = x.size(-1) // 2 * 2
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    b, s, h, d = x_rot.shape
    x_rot_complex = torch.view_as_complex(x_rot.float().reshape(b, s, h, d // 2, 2))
    freqs = freqs_cis[:s].unsqueeze(0).unsqueeze(2)
    x_out = x_rot_complex * freqs
    x_out_real = torch.view_as_real(x_out).reshape(b, s, h, rot_dim)
    return torch.cat((x_out_real, x_pass), dim=-1).to(torch.bfloat16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

with torch.no_grad():
    cfg = Config(path="/home/naklecha/.llama/checkpoints/Llama3.2-1B-Instruct", device=device)
    tokenizer, model = cfg.tokenizer, cfg.model

    formatted = [
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n" + p + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        for p in prompts
    ]
    batch_size = len(formatted)

    enc = [torch.tensor(tokenizer.encode(p, allowed_special="all"), dtype=torch.long) for p in formatted]
    max_len = max(len(t) for t in enc)
    pad_id = tokenizer.encode("<|end_of_text|>", allowed_special="all")[0]

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
    for i, t in enumerate(enc):
        input_ids[i, : t.numel()] = t.to(device)

    embed = torch.nn.Embedding(cfg.vocab_size, cfg.dim, _weight=model["tok_embeddings.weight"]).to(device)
    tokens_embed = embed(input_ids).to(torch.bfloat16)
    
    MAX_NEW = 200
    freqs_cis = get_freqs_cis(max_len + MAX_NEW, rope_theta=cfg.rope_theta, device=device)
    stop_token = tokenizer.encode("<|eot_id|>", allowed_special="all")[0]

    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    generated = torch.zeros(batch_size, dtype=torch.long, device=device)

    start = time.time()
    for step in range(MAX_NEW):
        token_start = time.time()
        active_indices = torch.where(active)[0]
        x = tokens_embed[active_indices]
        batch_size, seq_len, dim = x.shape
        for layer_index in range(cfg.n_layers):
            w_rms = model[f"layers.{layer_index}.attention_norm.weight"]
            x_norm = torch.nn.functional.rms_norm(x, w_rms.shape, w_rms, cfg.norm_eps)
            q = (x_norm @ model[f"layers.{layer_index}.attention.wq.weight"].T).view(batch_size, seq_len, cfg.n_heads, cfg.head_dim)
            k = (x_norm @ model[f"layers.{layer_index}.attention.wk.weight"].T).view(batch_size, seq_len, cfg.n_kv_heads, cfg.head_dim)
            v = (x_norm @ model[f"layers.{layer_index}.attention.wv.weight"].T).view(batch_size, seq_len, cfg.n_kv_heads, cfg.head_dim)
            q = apply_rope(q, freqs_cis)
            k = apply_rope(k, freqs_cis)
            if cfg.kv_mult > 1:
                k = k.repeat_interleave(cfg.kv_mult, dim=2)
                v = v.repeat_interleave(cfg.kv_mult, dim=2)
            mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q.permute(0, 2, 1, 3).reshape(-1, seq_len, cfg.head_dim),
                k.permute(0, 2, 1, 3).reshape(-1, seq_len, cfg.head_dim),
                v.permute(0, 2, 1, 3).reshape(-1, seq_len, cfg.head_dim),
                mask,
            )
            attn_out = attn_out.view(batch_size, cfg.n_heads, seq_len, cfg.head_dim).permute(0, 2, 1, 3).reshape(batch_size, seq_len, cfg.dim)
            x = x + attn_out @ model[f"layers.{layer_index}.attention.wo.weight"].T
            w_rms_ffn = model[f"layers.{layer_index}.ffn_norm.weight"]
            x_ffn_norm = torch.nn.functional.rms_norm(x, w_rms_ffn.shape, w_rms_ffn, cfg.norm_eps)
            w1, w2, w3 = (model[f"layers.{layer_index}.feed_forward.{w}.weight"] for w in ("w1", "w2", "w3"))
            ffn_out = (torch.nn.functional.silu(x_ffn_norm @ w1.T) * (x_ffn_norm @ w3.T)) @ w2.T
            x = x + ffn_out
        x = torch.nn.functional.rms_norm(x, model["norm.weight"].shape, model["norm.weight"], cfg.norm_eps)
        logits = (x[:, -1, :] @ model["output.weight"].T).float()
        next_toks = torch.argmax(logits, dim=-1)
        tokens_embed = torch.cat((tokens_embed, embed(next_toks).to(torch.bfloat16).unsqueeze(1)), dim=1)

        for i in range(batch_size):
            if active[i]:
                enc[i] = torch.cat((enc[i], next_toks[i:i+1]))
                generated[i] += 1
                if next_toks[i].item() == stop_token: active[i] = False
                    
        print(f"[TOTAL TOKENS GENERATED {(step+1)*50}] -- {time.time() - token_start:.3f}sec -- {1 / (time.time() - token_start) * 50:.1f}tps")
        if not active.any(): break

    elapsed = time.time() - start
    for i in range(batch_size):
        print(f"\n\nPROMPT {i+1}")
        print(tokenizer.decode(enc[i].tolist()))
        
    print("\n[GENERATION TIME]", elapsed)
    print("TOTAL TOKENS GENERATED", generated.sum().item())
    print("[TOKENS/S]", generated.sum().item() / elapsed)