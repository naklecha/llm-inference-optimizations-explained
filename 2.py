from prompts import prompts
from config import Config
from pathlib import Path
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

with torch.no_grad():
    config = Config(path=f"{Path.home()}/.llama/checkpoints/Llama3.2-1B-Instruct", device=device)
    tokenizer, model = config.tokenizer, config.model

    def get_freqs_cis(max_tokens):
        zero_to_one_split_into_64_parts = torch.tensor(range(32))/32
        freqs = 1.0 / (config.rope_theta ** zero_to_one_split_into_64_parts)
        freqs_for_each_token = torch.outer(torch.arange(max_tokens), freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)
        return freqs_cis

    def apply_rope(x, freqs_cis):
        rot_dim = config.head_dim // 2 * 2
        x_rotated, x_pass = x[..., :rot_dim], x[..., rot_dim:]
        x_rotated = x_rotated.float().view(x.shape[0], -1, rot_dim//2, 2)
        freqs_layer = freqs_cis[:x.shape[0], :rot_dim//2].unsqueeze(1)
        x_rotated = torch.view_as_real(torch.view_as_complex(x_rotated) * freqs_layer)
        x_rotated = x_rotated.view(x.shape)
        return torch.cat([x_rotated, x_pass], dim=-1).to(torch.bfloat16)
    
    MAX_TOKENS_TO_GENERATE = 500
    total_tokens_generated = 0
    
    formatted_prompts = []
    for p in prompts:
        prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n" + p + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        formatted_prompts.append(prompt)
        
    # running only on 5 prompts because it's so damn slow
    formatted_prompts = formatted_prompts[:5]
        
    generation_start = time.time()
    total_tokens_generated = 0
    for prompt_index, prompt in enumerate(formatted_prompts):
        tokens = torch.tensor(tokenizer.encode(prompt, allowed_special="all"))
        embedding_layer = torch.nn.Embedding(config.vocab_size, config.dim)
        embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])
        token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16)
        MAX_TOKENS = MAX_TOKENS_TO_GENERATE + len(tokens)
        freqs_cis = get_freqs_cis(MAX_TOKENS)
        for token_index in range(MAX_TOKENS_TO_GENERATE):
            token_start = time.time()
            seq_len = token_embeddings_unnormalized.size(0)
            final_embedding = token_embeddings_unnormalized
            for layer in range(config.n_layers):
                norm_weights = model[f"layers.{layer}.attention_norm.weight"]
                x_norm = torch.nn.functional.rms_norm(final_embedding, norm_weights.shape, norm_weights, config.norm_eps)
                q_layer = model[f"layers.{layer}.attention.wq.weight"]
                k_layer = model[f"layers.{layer}.attention.wk.weight"]
                v_layer = model[f"layers.{layer}.attention.wv.weight"]
                q = (x_norm @ q_layer.T).view(seq_len, config.n_heads, config.head_dim)
                k = (x_norm @ k_layer.T).view(seq_len, config.n_kv_heads, config.head_dim)
                v = (x_norm @ v_layer.T).view(seq_len, config.n_kv_heads, config.head_dim)
                q = apply_rope(q, freqs_cis)
                k = apply_rope(k, freqs_cis)
                if config.kv_mult > 1:
                    k = k.repeat_interleave(config.kv_mult, dim=1)
                    v = v.repeat_interleave(config.kv_mult, dim=1)
                attn_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
                attention_output = torch.nn.functional.scaled_dot_product_attention(q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2), attn_mask)
                attention_output = attention_output.permute(1, 0, 2).contiguous().view(seq_len, config.dim)
                w_layer = model[f"layers.{layer}.attention.wo.weight"]
                embedding_delta = attention_output @ w_layer.T
                embedding_after_edit = final_embedding + embedding_delta
                ffn_norm_weights = model[f"layers.{layer}.ffn_norm.weight"]
                embedding_after_edit_normalized = torch.nn.functional.rms_norm(embedding_after_edit, ffn_norm_weights.shape, ffn_norm_weights, config.norm_eps)
                w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
                w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
                w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
                output_after_feedforward = (torch.nn.functional.silu(embedding_after_edit_normalized @ w1.T) * (embedding_after_edit_normalized @ w3.T)) @ w2.T
                final_embedding = embedding_after_edit + output_after_feedforward
            final_embedding = torch.nn.functional.rms_norm(final_embedding, model["norm.weight"].shape, model["norm.weight"], config.norm_eps)
            logits = final_embedding[-1] @ model["output.weight"].T
            next_token = torch.argmax(logits, dim=-1)
            output = tokenizer.decode([next_token.item()])
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=0)
            token_embeddings_unnormalized = torch.cat([token_embeddings_unnormalized, embedding_layer(next_token.unsqueeze(0)).to(torch.bfloat16)], dim=0)
            token_end = time.time()
            print(f"[PROMPT {prompt_index+1}] [TOKEN {token_index+1}] {repr(output)} -- {token_end - token_start:.3f}sec -- {1 / (token_end - token_start):.1f}tps")
            prompt += output
            if output == "<|eot_id|>": break

        time_taken = time.time() - generation_start
        total_tokens_generated += len(tokens)
        print("[TOKENS]", tokens)
        
    print("[GENERATION TIME] ", time_taken)
    print("[OUTPUT TOKENS PER SECOND] ", total_tokens_generated / time_taken)