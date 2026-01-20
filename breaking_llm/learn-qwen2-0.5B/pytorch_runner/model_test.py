import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.as_posix())

import os
import random
import numpy as np
import json
import torch

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from qwen2_model_def import Qwen2ForCausalLM, Qwen2DecoderLayer, Qwen2MLP

from model import MLP, ModelConfig, Attention

cur_dir = Path(__file__).parent
data_dir = cur_dir.parent / "data"

def set_all_seeds(seed=42):
    """设置所有随机数种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_my_model_config() -> ModelConfig:
    return ModelConfig()

def load_transformers_qwen2_05_instruct() -> tuple[Qwen2ForCausalLM, Qwen2TokenizerFast]:
    from safetensors.torch import load_file

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
        tokenizer_file=str(model_path / "tokenizer.json"),
    )
    return model, tokenizer

def test_mlp(qwen_model: Qwen2ForCausalLM, my_model_config: ModelConfig):
    first_layer:Qwen2DecoderLayer = qwen_model.model.layers[0]
    first_mlp:Qwen2MLP = first_layer.mlp

    my_mlp = MLP(my_model_config).to(qwen_model.device)
    my_mlp.load_state_dict(first_mlp.state_dict())
    my_mlp = my_mlp.to(dtype=qwen_model.dtype, device=qwen_model.device)

    x = torch.randn(2, 4, my_model_config.hidden_size).to(dtype=qwen_model.dtype, device=qwen_model.device)
    qwen_y = first_mlp(x)
    my_y = my_mlp(x)

    assert torch.allclose(qwen_y, my_y, atol=1e-6)
    print("MLP test passed!")

def test_attention_without_mask_and_cache(qwen_model: Qwen2ForCausalLM, my_model_config: ModelConfig):
    first_layer:Qwen2DecoderLayer = qwen_model.model.layers[0]
    first_attention = first_layer.self_attn
    first_attention.config._attn_implementation = "eager"

    my_attention = Attention(my_model_config, 0)
    my_attention.load_state_dict(first_attention.state_dict())
    my_attention = my_attention.to(dtype=qwen_model.dtype, device=qwen_model.device)

    batch_size = 4
    seq_len = 128
    hidden_size = my_model_config.hidden_size
    head_dim = my_attention.head_dim
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32).to(device=qwen_model.device)
    position_embeddings = (
        torch.randn(1, seq_len, head_dim, dtype=torch.float32).to(device=qwen_model.device),
        torch.randn(1, seq_len, head_dim, dtype=torch.float32).to(device=qwen_model.device),
    )

    qwen_attention_output, qwen_attention_weights = first_attention.forward(hidden_states, position_embeddings, attention_mask=None)
    my_attention_output, my_attention_weights = my_attention.forward(hidden_states, position_embeddings, attention_mask=None)

    assert torch.allclose(qwen_attention_output, my_attention_output, atol=1e-5)
    if qwen_attention_weights is not None:
        assert torch.allclose(qwen_attention_weights, my_attention_weights, atol=1e-5)
    print("Attention test passed")

if __name__ == "__main__":
    set_all_seeds(42)
    qwen_model, qwen_tokenizer = load_transformers_qwen2_05_instruct()
    my_model_config = load_my_model_config()
    test_mlp(qwen_model, my_model_config)
    test_attention_without_mask_and_cache(qwen_model, my_model_config)

    