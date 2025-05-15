import torch

path = "/home/naklecha/.llama/checkpoints/Llama3.2-1B-Instruct"
model = torch.load(f"{path}/consolidated.00.pth", map_location="cpu")

import numpy as np

weights = {}
for k, v in model.items():
    if v.dtype == torch.bfloat16:
        v = v.to(torch.float32)
    weights[k] = v.cpu().numpy()

np.savez("model-weights.npz", **weights)