import time
import shutil
from pathlib import Path
from transformers.generation.streamers import BaseStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer

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
