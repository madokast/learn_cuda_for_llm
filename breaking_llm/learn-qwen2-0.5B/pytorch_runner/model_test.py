import sys
sys.path.append("..")

from pathlib import Path
import torch

from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2DecoderLayer, Qwen2MLP
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

from model import MLP, ModelConfig

cur_dir = Path(__file__).parent
data_dir = cur_dir.parent / "data"

def load_my_model_config() -> ModelConfig:
    return ModelConfig()

def load_transformers_qwen2_05_instruct() -> tuple[Qwen2ForCausalLM, Qwen2TokenizerFast]:
    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(__file__).parent.parent / "qwen2-0.5B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto"
    )
    print(f"Model loaded to {model.device} using dtype {model.dtype}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
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

if __name__ == "__main__":
    qwen_model, qwen_tokenizer = load_transformers_qwen2_05_instruct()
    my_model_config = load_my_model_config()
    test_mlp(qwen_model, my_model_config)

    