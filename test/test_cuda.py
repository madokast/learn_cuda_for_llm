import torch
print("CUDA 可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA 设备数量:", torch.cuda.device_count())
    print("当前设备索引:", torch.cuda.current_device())
    print("设备名称:", torch.cuda.get_device_name(0))
    print("CuDNN 版本:", torch.backends.cudnn.version())
    # 测试 CUDA 计算
    x = torch.tensor([1.0, 2.0]).cuda()
    y = torch.tensor([3.0, 4.0]).cuda()
    print("CUDA 计算结果:", x + y)
else:
    print("没有可用的 CUDA 设备，PyTorch 将使用 CPU")

    