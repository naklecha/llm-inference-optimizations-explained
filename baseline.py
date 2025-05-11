from prompts import prompts
import time
import vllm

model = vllm.LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    device="cuda",
    dtype="bfloat16",
)

start = time.time()

result = model.generate(prompts, sampling_params=vllm.SamplingParams(temperature=0.0, max_tokens=500))
print(result)

time_taken = time.time() - start

total_tokens = sum([len(r.outputs[0].token_ids) for r in result])
print("\n\n[GENERATION TIME] ", time_taken)
print("[OUTPUT TOKENS] ", total_tokens)
print("[OUTPUT TOKENS PER SECOND] ", total_tokens / time_taken)