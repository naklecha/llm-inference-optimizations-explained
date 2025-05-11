from config import Config
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

with torch.no_grad():
    config = Config(
        # "/Users/naklecha/.llama/checkpoints/Llama3.2-1B-Instruct"
        path="/home/naklecha/.llama/checkpoints/Llama3.2-1B-Instruct",
        device=device
    )

    tokenizer = config.tokenizer
    model = config.model

    prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is the meaning of life?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    tokens = tokenizer.encode(prompt, allowed_special="all")
    tokens = torch.tensor(tokens)
    MAX_TOKENS_TO_GENERATE = 100
    MAX_TOKENS = MAX_TOKENS_TO_GENERATE + len(tokens)
    print("Input tokens:", tokens)

    embedding_layer = torch.nn.Embedding(config.vocab_size, config.dim)
    embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])
    token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16)

    zero_to_one_split_into_64_parts = torch.tensor(range(32))/32
    freqs = 1.0 / (config.rope_theta ** zero_to_one_split_into_64_parts)
    freqs_for_each_token = torch.outer(torch.arange(MAX_TOKENS), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)

    for layer in range(config.n_layers):
        model[f"layers.{layer}.attention.wq.weight"] =  model[f"layers.{layer}.attention.wq.weight"].view(config.n_heads, config.head_dim, config.dim)
        model[f"layers.{layer}.attention.wk.weight"] =  model[f"layers.{layer}.attention.wk.weight"].view(config.n_kv_heads, config.head_dim, config.dim)
        model[f"layers.{layer}.attention.wv.weight"] =  model[f"layers.{layer}.attention.wv.weight"].view(config.n_kv_heads, config.head_dim, config.dim)
    
    generation_start = time.time()
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
                q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
                q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
                q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis[:q_per_token.shape[0]])
                q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)
                k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
                k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
                k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis[:k_per_token.shape[0]])
                k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)
                qk_per_token = q_per_token_rotated @ k_per_token_rotated.T/(128)**0.5
                mask = torch.full((len(token_embeddings_unnormalized), len(token_embeddings_unnormalized)), float("-inf"))
                mask = torch.triu(mask, diagonal=1)
                qk_per_token_after_masking = qk_per_token + mask
                qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1, dtype=torch.bfloat16)
                qkv_attention = qk_per_token_after_masking_after_softmax @ v_per_token
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
        print(f"[PROMPT TOKEN {token_index+1}] {repr(output)} -- {time.time() - token_start:.3f}sec -- {1 / (time.time() - token_start):.1f}tps")
        if output == "<|eot_id|>": break
    time_taken = time.time() - generation_start
    
    print("TOTAL TOKENS GENERATED", token_index)
    print("[GENERATION TIME] ", time_taken)
    print("[OUTPUT TOKENS PER SECOND] ", token_index / time_taken)
    print("[END] Generation complete")
    print("[TOKENS]", tokens)