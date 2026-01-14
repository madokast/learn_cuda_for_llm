import sys
sys.path.append("..")

import json
import torch
from pathlib import Path
from safetensors.torch import load_file

# .venv/lib/python3.12/site-packages/transformers/models/qwen2
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

from transformers.tokenization_utils_base import BatchEncoding
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.streamers import TextStreamer

from model_def import Qwen2ForCausalLM

model_path = Path(__file__).parent.parent / "qwen2-0.5B-Instruct"

# 加载模型配置
with open(model_path / "config.json", "r") as f:
    config:Qwen2Config = Qwen2Config(**json.load(f))

# 创建空模型架构
model:Qwen2ForCausalLM = Qwen2ForCausalLM(config)

# 加载权重文件
weights = load_file(model_path / "model.safetensors")

# 将权重分配到模型（设置strict=False，忽略缺失的lm_head.weight）
model.load_state_dict(weights, strict=False)

# 手动将embed_tokens.weight赋值给lm_head.weight（因为tie_word_embeddings=True）
model.lm_head.weight = model.model.embed_tokens.weight

# 移动模型到设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# 加载tokenizer
tokenizer:Qwen2TokenizerFast = Qwen2TokenizerFast(
    vocab_file=str(model_path / "vocab.json"),
    merges_file=str(model_path / "merges.txt"),
    tokenizer_file=str(model_path / "tokenizer.json"),
)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]


text:str = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Give me a short introduction to large language model.<|im_end|>
<|im_start|>assistant\n"""

# model_inputs = {"input_ids": tensor, "attention_mask": tensor}
model_inputs:BatchEncoding = tokenizer([text], return_tensors="pt").to(device)

# 创建流式器
streamer = TextStreamer(tokenizer, skip_prompt=True)

with open(model_path / "generation_config.json") as f:
    generation_config = GenerationConfig(**json.load(f))
_ = model.generate(
    **model_inputs,
    generation_config=generation_config,
    max_new_tokens=512,
    streamer=streamer
)
