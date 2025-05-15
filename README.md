# implementing llm inference and making it go faster
in this repository, i'm going to implement increasingly complex llm inference optimizations
to understand the basics of how llms work, refer to my other respository where i implement llama3 from scratch and explain how llms work on matrix multiplication at a time:
https://github.com/naklecha/llama3-from-scratch; this repo of mine has like 15k stars, kinda wild!

(this repo is currently wip)

note: this is not production quality code, and it will never be, it's just for educational purposes. i like single file, no functions, no classes, codebases that are easy to understand. also, writing code this way is alot more aesthetic.

#### what i have so far, in order of increasing complexity:
- 0.py (15 tokens per second) similar to my llama3-from-scratch repo
- 1.py (15 tokens per second) same as 0.py but for multiple prompts
- 2.py (116 tokens per second) uses batch matrix multiplications for attention computation (all heads at once)
- 3.py (342 tokens per second) multiple prompts simultaneously with improved matrix operations and parallel token generation
- 4.py (7160 tokens per second) added kv caching, proper batch processing (removing completed prompts from the batch)
<br>.
<br>.
<br>.<br>
one day this will be 50.py & faster than vllm
- baseline.py (10514 tokens per second, single prompt) using vllm, 50 prompts at once, 500 tokens generated each

#### deets:
- hardware right now: single 4090
- model: llama3.2-1b-instruct
- batch size: 50 prompts at once
- if you don't want to run the code yourself, you can look at the outputs in the outputs folder.

