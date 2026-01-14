"""
下载 qwen2-0.5B-Instruct
export HF_ENDPOINT=https://hf-mirror.com
"""

import time
import shutil
from pathlib import Path
from transformers.generation.streamers import BaseStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
mdk@Xiaoxinpro16:~/learn_cuda_for_llm/breaking_llm/qwen2-0.5B-Instruct$ ll
total 980484
drwxrwxr-x 2 mdk mdk      4096 Jan 13 21:30 ./
drwxrwxr-x 3 mdk mdk      4096 Jan 13 21:32 ../
-rw-rw-r-- 1 mdk mdk       605 Jan 13 21:29 added_tokens.json
-rw-rw-r-- 1 mdk mdk      2507 Jan 13 21:29 chat_template.jinja
-rw-rw-r-- 1 mdk mdk      1227 Jan 13 21:29 config.json
-rw-rw-r-- 1 mdk mdk       242 Jan 13 21:29 generation_config.json
-rw-rw-r-- 1 mdk mdk   1671853 Jan 13 21:29 merges.txt
-rw-rw-r-- 1 mdk mdk 988097824 Jan 13 21:29 model.safetensors
-rw-rw-r-- 1 mdk mdk       613 Jan 13 21:29 special_tokens_map.json
-rw-rw-r-- 1 mdk mdk  11421896 Jan 13 21:29 tokenizer.json
-rw-rw-r-- 1 mdk mdk      4686 Jan 13 21:29 tokenizer_config.json
-rw-rw-r-- 1 mdk mdk   2776833 Jan 13 21:29 vocab.json

### added_tokens.json
  定义了模型额外添加的特殊标记及其对应的ID映射，包括对话标记、工具调用标记和多模态相关标记等。

### chat_template.jinja
  定义了聊天对话的格式模板，规定了系统提示、用户输入、助手输出和工具调用等的格式规范。

### config.json
  模型的核心架构配置文件，包含模型类型、隐藏层大小、注意力头数、最大序列长度等关键参数。

### generation_config.json
  文本生成配置文件，定义了模型生成响应时的采样策略、温度、top_k、top_p和重复惩罚等参数。

### merges.txt
  BPE（字节对编码）分词器的合并规则文件，用于将文本分割为模型可理解的标记。

### model.safetensors
  模型的核心权重文件，采用安全张量格式存储，包含了模型训练后的所有参数，大小约988MB。

### special_tokens_map.json
  特殊标记的映射关系文件，用于模型理解和处理各种特殊标记。

### tokenizer.json
  完整的分词器配置文件，包含词汇表、合并规则和所有分词器设置，用于文本的编码和解码。

### tokenizer_config.json
  分词器的基本配置文件，指定了分词器类、特殊标记和其他分词相关设置。

### vocab.json
  分词器的词汇表文件，包含所有可识别的标记及其对应的ID映射，用于文本的标记化处理。
"""

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# 模型保存目录
model_dir = Path(__file__).parent / "qwen2-0.5B-Instruct"
# 下载模型地址
cache_dir = Path(__file__).parent / "temp"

model = AutoModelForCausalLM.from_pretrained(
    model_dir if model_dir.exists() else model_name,
    dtype="auto",
    device_map="auto",
    cache_dir=cache_dir
)
tokenizer = AutoTokenizer.from_pretrained(
    model_dir if model_dir.exists() else model_name,
    cache_dir=cache_dir,
    fix_mistral_regex=True
)
print(model.device)

if not model_dir.exists():
    print("保存模型到", model_dir)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    shutil.rmtree(cache_dir)
    print("模型保存完成")


prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 创建流式器
class MyStreamer(BaseStreamer):
    def __init__(self, tokenizer):
        self.in_promot = True
        self.tokenizer = tokenizer
        self.token_count = 0
        self.time_to_first_token = 0
        self.start_time = time.time()

    def put(self, value):
        if self.in_promot:
            self.in_promot = False
            self.start_time = time.time()
            return
        if self.token_count == 0:
            self.time_to_first_token = time.time() - self.start_time
        self.token_count += len(value)
        token = self.tokenizer.decode(value, skip_special_tokens=True)
        if token != "": 
            print(token, end="", flush=True)

    def end(self):
        print(f"\n\nTime to first token: {self.time_to_first_token:.4f} seconds")
        print(f"Tokens per second: {self.token_count / self.time_to_first_token:.4f}")
        print()

streamer = MyStreamer(tokenizer)

_ = model.generate(
    **model_inputs,
    max_new_tokens=512,
    streamer=streamer
)

"""
Time to first token: 1.0533 seconds
Tokens per second: 185.1280
"""
