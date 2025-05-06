from tokenizer import get_tokenizer
import torch
import json
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

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
    head_dim = dim // n_heads
    kv_multiple = n_heads // n_kv_heads

    prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nWhat is the meaning of life?Generate 1000 different possible answers.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    tokens = torch.tensor(tokenizer.encode(prompt, allowed_special="all"))
    prompt_split_as_tokens = [tokenizer.decode([token.item()]) for token in tokens]
    MAX_TOKENS_TO_GENERATE = 1000
    MAX_TOKENS = MAX_TOKENS_TO_GENERATE + len(prompt_split_as_tokens)

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
        seq_len = token_embeddings_unnormalized.size(0)
        final_embedding = token_embeddings_unnormalized
        for layer in range(n_layers):
            x_norm = rms_norm(final_embedding, model[f"layers.{layer}.attention_norm.weight"])
            q_layer = model[f"layers.{layer}.attention.wq.weight"]
            k_layer = model[f"layers.{layer}.attention.wk.weight"]
            v_layer = model[f"layers.{layer}.attention.wv.weight"]
            q = torch.matmul(x_norm, q_layer.T)
            k = torch.matmul(x_norm, k_layer.T)
            v = torch.matmul(x_norm, v_layer.T)
            q = q.view(seq_len, n_heads, head_dim)
            k = k.view(seq_len, n_kv_heads, head_dim)
            v = v.view(seq_len, n_kv_heads, head_dim)
            rot_dim = head_dim // 2 * 2
            q_rotated, q_pass = q[..., :rot_dim], q[..., rot_dim:]
            k_rotated, k_pass = k[..., :rot_dim], k[..., rot_dim:]
            q_rotated = q_rotated.float().view(seq_len, -1, rot_dim//2, 2)
            k_rotated = k_rotated.float().view(seq_len, -1, rot_dim//2, 2)
            freqs_layer = freqs_cis[:seq_len, :rot_dim//2].unsqueeze(1)
            q_rotated = torch.view_as_real(torch.view_as_complex(q_rotated) * freqs_layer)
            k_rotated = torch.view_as_real(torch.view_as_complex(k_rotated) * freqs_layer)
            q = torch.cat([q_rotated.view(seq_len, -1, rot_dim), q_pass], dim=-1)
            k = torch.cat([k_rotated.view(seq_len, -1, rot_dim), k_pass], dim=-1)
            if kv_multiple > 1:
                k = k.repeat_interleave(kv_multiple, dim=1)
                v = v.repeat_interleave(kv_multiple, dim=1)
            q = q.permute(1, 0, 2)
            k = k.permute(1, 0, 2)
            v = v.permute(1, 0, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
            scores = scores + mask
            attention = torch.softmax(scores, dim=-1).to(torch.bfloat16)
            attention_output = torch.matmul(attention, v)
            attention_output = attention_output.permute(1, 0, 2).contiguous().view(seq_len, dim)
            w_layer = model[f"layers.{layer}.attention.wo.weight"]
            embedding_delta = torch.matmul(attention_output, w_layer.T)
            embedding_after_edit = final_embedding + embedding_delta
            embedding_after_edit_normalized = rms_norm(embedding_after_edit, model[f"layers.{layer}.ffn_norm.weight"])
            w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
            w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
            w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
            output_after_feedforward = torch.matmul(torch.nn.functional.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T)
            final_embedding = embedding_after_edit + output_after_feedforward
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
    print("[GENERATION TIME] ", time_taken)
    print("[OUTPUT TOKENS PER SECOND] ", (len(tokens)-len(prompt_split_as_tokens)) / time_taken)
    print("[END] Generation complete")
    print("[TOKENS]", tokens)