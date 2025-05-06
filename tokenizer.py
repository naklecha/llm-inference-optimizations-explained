from pathlib import Path
import tiktoken, tiktoken.load

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