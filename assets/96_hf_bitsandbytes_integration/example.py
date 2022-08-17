import torch
import torch.nn as nn

from bitsandbytes.nn import Linear8bitLt

# Utility function

def get_model_memory_footprint(model):
    r"""
        Partially copied and inspired from: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822/2
    """
    return sum([param.nelement() * param.element_size() for param in model.parameters()])

# Main script

fp16_model = nn.Sequential(
    nn.Linear(64, 64),
    nn.Linear(64, 64)
).to(torch.float16)

# Train and save your model!

torch.save(fp16_model.state_dict(), "model.pt")

# Define your int8 model!

int8_model = nn.Sequential(
    Linear8bitLt(64, 64, has_fp16_weights=False),
    Linear8bitLt(64, 64, has_fp16_weights=False)
)

int8_model.load_state_dict(torch.load("model.pt"))
int8_model = int8_model.to(0) # Quantization happens here

input_ = torch.randn(8, 64, dtype=torch.float16)
hidden_states = int8_model(input_.to(0))

mem_int8 = get_model_memory_footprint(int8_model)
mem_fp16 = get_model_memory_footprint(fp16_model)

print(f"Relative difference: {mem_fp16/mem_int8}")