from prompts import prompts
import time
import vllm

model = vllm.LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    device="cuda",
    dtype="bfloat16",
)

start = time.time()

result = model.generate(prompts, sampling_params=vllm.SamplingParams(temperature=0.0, max_tokens=1000))
print(result)

time_taken = time.time() - start

print("[GENERATION TIME] ", time_taken)
print("[OUTPUT TOKENS PER SECOND] ", sum([len(r.outputs[0].token_ids) for r in result]) / time_taken)