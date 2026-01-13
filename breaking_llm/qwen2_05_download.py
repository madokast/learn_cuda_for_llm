import shutil
from pathlib import Path
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

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)