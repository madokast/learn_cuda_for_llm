from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from safetensors.torch import load_file

model_path = Path(__file__).parent / "qwen2-0.5B-Instruct"

# 加载模型配置
config = AutoConfig.from_pretrained(model_path)

# 创建空模型架构
model = AutoModelForCausalLM.from_config(config)

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
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# 手动实现token生成
max_new_tokens = 512
generated_ids = model_inputs.input_ids

# while循环生成token
while generated_ids.shape[1] < model_inputs.input_ids.shape[1] + max_new_tokens:
    # 调用forward方法获取logits
    with torch.no_grad():
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