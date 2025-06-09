try:
    import torchvision 
    print("torchvision is available")
except ImportError:
    print("torchvision is NOT available")

try:
    import torch
    print("torch is available")
except ImportError:
    print("torch is NOT available")

if torch.cuda.is_available():
	print(torch.cuda.get_device_name(0))
else:
	print("No CUDA device available")
