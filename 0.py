from tokenizer import get_tokenizer
import torch
import json
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
# disable gradient calculation, for inference (and speed)

with torch.no_grad():
    print("Loading model and tokenizer...")
    load_start = time.time()

    # PATH = "/Users/naklecha/.llama/checkpoints/Llama3.2-1B-Instruct"
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
    multiple_of = config["multiple_of"]
    ffn_dim_multiplier = config["ffn_dim_multiplier"]
    norm_eps = config["norm_eps"]
    rope_theta = torch.tensor(config["rope_theta"])
    kv_multiple = n_heads // n_kv_heads
    head_dim = dim // n_heads

    prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nthe answer to the ultimate question, the universe and everything is?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    tokens = tokenizer.encode(prompt, allowed_special="all")
    tokens = torch.tensor(tokens)
    prompt_split_as_tokens = [tokenizer.decode([token.item()]) for token in tokens]
    MAX_TOKENS_TO_GENERATE = 1000
    MAX_TOKENS = MAX_TOKENS_TO_GENERATE + len(prompt_split_as_tokens)
    print("Input tokens:", tokens)

    embedding_layer = torch.nn.Embedding(vocab_size, dim)
    embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])
    token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16)

    zero_to_one_split_into_64_parts = torch.tensor(range(32))/32
    freqs = 1.0 / (rope_theta ** zero_to_one_split_into_64_parts)
    freqs_for_each_token = torch.outer(torch.arange(MAX_TOKENS), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)
    rms_norm = lambda tensor, norm_weights: (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) * norm_weights

    generation_start = time.time()

    for i in range(MAX_TOKENS_TO_GENERATE):
        token_start = time.time()
        final_embedding = token_embeddings_unnormalized
        for layer in range(n_layers):
            qkv_attention_store = []
            layer_embedding_norm = rms_norm(final_embedding, model[f"layers.{layer}.attention_norm.weight"])
            q_layer = model[f"layers.{layer}.attention.wq.weight"]
            q_layer = q_layer.view(n_heads, head_dim, dim)
            k_layer = model[f"layers.{layer}.attention.wk.weight"]
            k_layer = k_layer.view(n_kv_heads, head_dim, dim)
            v_layer = model[f"layers.{layer}.attention.wv.weight"]
            v_layer = v_layer.view(n_kv_heads, head_dim, dim)
            w_layer = model[f"layers.{layer}.attention.wo.weight"]
            for head in range(n_heads):
                q_layer_head = q_layer[head]
                k_layer_head = k_layer[head//kv_multiple]
                v_layer_head = v_layer[head//kv_multiple]
                q_per_token = torch.matmul(layer_embedding_norm, q_layer_head.T)
                k_per_token = torch.matmul(layer_embedding_norm, k_layer_head.T)
                v_per_token = torch.matmul(layer_embedding_norm, v_layer_head.T)
                q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
                q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
                q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis[:q_per_token.shape[0]])
                q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)
                k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
                k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
                k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis[:k_per_token.shape[0]])
                k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)
                qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(128)**0.5
                mask = torch.full((len(token_embeddings_unnormalized), len(token_embeddings_unnormalized)), float("-inf"))
                mask = torch.triu(mask, diagonal=1)
                qk_per_token_after_masking = qk_per_token + mask
                qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1, dtype=torch.bfloat16)
                qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
                qkv_attention_store.append(qkv_attention)
            stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
            w_layer = model[f"layers.{layer}.attention.wo.weight"]
            embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T)
            embedding_after_edit = final_embedding + embedding_delta
            embedding_after_edit_normalized = rms_norm(embedding_after_edit, model[f"layers.{layer}.ffn_norm.weight"])
            w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
            w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
            w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
            output_after_feedforward = torch.matmul(
                torch.functional.F.silu(
                    torch.matmul(embedding_after_edit_normalized, w1.T)
                ) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T
            )
            final_embedding = embedding_after_edit+output_after_feedforward
        final_embedding = rms_norm(final_embedding, model["norm.weight"])
        logits = torch.matmul(final_embedding[-1], model["output.weight"].T)
        next_token = torch.argmax(logits, dim=-1)
        output = tokenizer.decode([next_token.item()])
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=0)
        token_embeddings_unnormalized = torch.cat([token_embeddings_unnormalized, embedding_layer(next_token.unsqueeze(0)).to(torch.bfloat16)], dim=0)
        print(f"[TOKEN {i+1}] {repr(output)} -- {time.time() - token_start:.3f}s")
        prompt += output
        if output == "<|eot_id|>": break

    time_taken = time.time() - generation_start
    print("TOTOAL TOKENS GENERATED", len(tokens)-len(prompt_split_as_tokens))
    print("[GENERATION TIME] ", time_taken)
    print("[OUTPUT TOKENS PER SECOND] ", (len(tokens)-len(prompt_split_as_tokens)) / time_taken)
    print("[END] Generation complete")
    print("[TOKENS]", tokens)