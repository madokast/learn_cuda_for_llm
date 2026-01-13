from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = Path(__file__).parent / "qwen2-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)

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

# 手动实现token生成
max_new_tokens = 512
generated_ids = model_inputs.input_ids

# while循环生成token
while generated_ids.shape[1] < model_inputs.input_ids.shape[1] + max_new_tokens:
    # 调用forward方法获取logits
    outputs = model(generated_ids)
    logits = outputs.logits
    
    # 获取最后一个token的logits
    next_token_logits = logits[:, -1, :]
    
    # 计算概率分布
    probs = torch.softmax(next_token_logits, dim=-1)
    
    # 采样下一个token（greedy采样，取概率最高的token）
    next_token_id = torch.argmax(probs, dim=-1, keepdim=True)
    
    # 将新token添加到生成序列
    generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
    
    # 检查是否生成了终止token
    if next_token_id.item() == tokenizer.eos_token_id:
        break

# 提取新生成的token
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# 解码并输出结果
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)