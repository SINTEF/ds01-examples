"""
Prints whether you have access to a CUDA device with PyTorch.

Quickstart:
$ mamba ceate --name pytorch python=3.11
$ mamba activate pytorch
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

References:
- https://stackoverflow.com/a/53374933/21124232
- https://pytorch.org/get-started/locally/
"""

import torch

# setting device on GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()

# Additional Info when using cuda
if device.type == "cuda":
    print(torch.cuda.get_device_name(0))
    print("Memory Usage:")
    print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
    print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")
