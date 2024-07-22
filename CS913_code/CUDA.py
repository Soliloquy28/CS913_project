import torch

# 获取当前设备的可用内存
def get_free_memory():
    torch.cuda.empty_cache()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    free_memory = total_memory - allocated_memory
    return free_memory / (1024 ** 3)  # 转换为GB

# 打印可用内存
print(f"当前可用CUDA内存: {get_free_memory():.2f} GB")
print(torch.cuda.device_count())