# 学习 qwen2-0.5B-Instruct 架构

- qwen2_05_download.py 从 huggingface 获取模型
- transforms_run.py 利用 transforms 运行模型，最简化的代码
- transforms_run_manual.py 利用 transforms 实例化模型，但是推理部分暴露了循环。不知道为什么效率很低

- deep_transforms_runner 文件夹，将 transforms 具体的 Qwen2Config、Qwen2TokenizerFast、Qwen2ForCausalLM 暴露了出来

- pytorch_runner 文件夹，去除 transforms 框架，使用 pytorch 框架进行推理

