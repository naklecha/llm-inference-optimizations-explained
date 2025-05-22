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
        x_float = x.float()
        x_float = x_float.view(x.shape[0], -1, 2)
        x_float_as_complex_numbers = torch.view_as_complex(x_float)
        x_float_as_complex_numbers_rotated = x_float_as_complex_numbers * freqs_cis[:x.shape[0]]
        x_float_as_complex_numbers_rotated = torch.view_as_real(x_float_as_complex_numbers_rotated)
        x_float_rotated = x_float_as_complex_numbers_rotated.view(x.shape)
        return x_float_rotated.to(torch.bfloat16)

    for layer in range(config.n_layers):
        model[f"layers.{layer}.attention.wq.weight"] =  model[f"layers.{layer}.attention.wq.weight"].view(config.n_heads, config.head_dim, config.dim)
        model[f"layers.{layer}.attention.wk.weight"] =  model[f"layers.{layer}.attention.wk.weight"].view(config.n_kv_heads, config.head_dim, config.dim)
        model[f"layers.{layer}.attention.wv.weight"] =  model[f"layers.{layer}.attention.wv.weight"].view(config.n_kv_heads, config.head_dim, config.dim)
    
    MAX_TOKENS_TO_GENERATE = 100
    total_tokens_generated = 0
    
    formatted_prompts = []
    for p in prompts:
        prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n" + p + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        formatted_prompts.append(prompt)
        
    # running only on 5 prompts because it's so damn slow
    formatted_prompts = formatted_prompts[:5]
    
    generation_start = time.time()
    for prompt_index, prompt in enumerate(formatted_prompts):
        tokens = torch.tensor(tokenizer.encode(prompt, allowed_special="all"))
        embedding_layer = torch.nn.Embedding(config.vocab_size, config.dim)
        embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])
        token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16)
        MAX_TOKENS = MAX_TOKENS_TO_GENERATE + len(tokens)
        freqs_cis = get_freqs_cis(MAX_TOKENS)
        for token_index in range(MAX_TOKENS_TO_GENERATE):
            token_start = time.time()
            final_embedding = token_embeddings_unnormalized
            for layer in range(config.n_layers):
                qkv_attention_store = []
                norm_weights = model[f"layers.{layer}.attention_norm.weight"]
                layer_embedding_norm = torch.nn.functional.rms_norm(final_embedding, normalized_shape=norm_weights.shape, weight=norm_weights, eps=config.norm_eps)
                q_layer = model[f"layers.{layer}.attention.wq.weight"]
                k_layer = model[f"layers.{layer}.attention.wk.weight"]
                v_layer = model[f"layers.{layer}.attention.wv.weight"]
                w_layer = model[f"layers.{layer}.attention.wo.weight"]
                for head in range(config.n_heads):
                    q_layer_head = q_layer[head]
                    k_layer_head = k_layer[head//config.kv_mult]
                    v_layer_head = v_layer[head//config.kv_mult]
                    q_per_token = layer_embedding_norm @ q_layer_head.T
                    k_per_token = layer_embedding_norm @ k_layer_head.T
                    v_per_token = layer_embedding_norm @ v_layer_head.T
                    q_per_token_rotated = apply_rope(q_per_token, freqs_cis)
                    k_per_token_rotated = apply_rope(k_per_token, freqs_cis)
                    attn_mask = torch.triu(torch.full((q_per_token_rotated.shape[0], q_per_token_rotated.shape[0]), float("-inf")), diagonal=1)
                    qkv_attention = torch.nn.functional.scaled_dot_product_attention(q_per_token_rotated, k_per_token_rotated, v_per_token, attn_mask)
                    qkv_attention_store.append(qkv_attention)
                stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
                w_layer = model[f"layers.{layer}.attention.wo.weight"]
                embedding_delta = stacked_qkv_attention @ w_layer.T
                embedding_after_edit = final_embedding + embedding_delta
                ffn_norm_weights = model[f"layers.{layer}.ffn_norm.weight"]
                embedding_after_edit_normalized = torch.nn.functional.rms_norm(embedding_after_edit, ffn_norm_weights.shape, ffn_norm_weights, config.norm_eps)
                w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
                w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
                w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
                output_after_feedforward = (torch.functional.F.silu(embedding_after_edit_normalized @ w1.T) * (embedding_after_edit_normalized @ w3.T)) @ w2.T
                final_embedding = embedding_after_edit+output_after_feedforward
            final_embedding = torch.nn.functional.rms_norm(final_embedding, normalized_shape=model["norm.weight"].shape, weight=model["norm.weight"], eps=config.norm_eps)
            logits = final_embedding[-1] @ model["output.weight"].T
            next_token = torch.argmax(logits, dim=-1)
            output = tokenizer.decode([next_token.item()])
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=0)
            token_embeddings_unnormalized = torch.cat([token_embeddings_unnormalized, embedding_layer(next_token.unsqueeze(0)).to(torch.bfloat16)], dim=0)
            prompt += output
            total_tokens_generated += 1
            print(f"[PROMPT {prompt_index+1} TOKEN {token_index+1}] {repr(output)} -- {time.time() - token_start:.3f}sec -- {1 / (time.time() - token_start):.1f}tps")
            print(f"[TOTAL TOKENS GENERATED]: {total_tokens_generated}")
            print("\n")
            if output == "<|eot_id|>": break
    time_taken = time.time() - generation_start

    print("TOTAL TOKENS GENERATED", total_tokens_generated)
    print("[GENERATION TIME] ", time_taken)
    print("[AVG TOKENS PER SECOND] ", total_tokens_generated / time_taken)
    print("[END] Generation complete")
    print("[TOKENS]", tokens)