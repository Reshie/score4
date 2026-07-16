import torch
print("CUDA利用可能か:", torch.cuda.is_available())
print("デバイス名:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "無し")
