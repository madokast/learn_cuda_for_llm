import sys
sys.path.append("..")

import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from rope import apply_rotary_pos_emb, RotaryEmbedding
from kv_cache import Cache

@dataclass
class ModelConfig:
    """模型配置信息，默认值来自 Qwen2-0.5B 模型"""
    vocab_size:int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    max_position_embeddings:int = 32768
    rope_theta: float = 1000000.0


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # config.hidden_act is silu
        self.act_fn = F.silu

    def forward(self, x):
        # x: linear hidden_size -> intermediate_size
        #    silu
        # x: linear hidden_size -> intermediate_size
        # *
        # r: linear intermediate_size -> hidden_size
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads # 896 / 14 = 64
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads # 14 / 2 = 7
        self.scaling = self.head_dim**-0.5 # 64**-0.5 = 1/8 = 0.125
        self.attention_dropout = 0.0 # config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True) # 896 -> 14*64(896)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True) # 896 -> 2*64(128)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True) # 896 -> 2*64(128)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False) # 14*64(896) -> 896

        # 所有层都是 full_attention 不是 sliding_attention
        # sliding_attention通过固定大小的滑动窗口（如512个token）来限制注意力范围，每个位置只关注窗口内的邻居，大幅降低了计算和内存开销。
        # 虽然单层只能捕获局部信息，但通过多层堆叠（如12层），信息可以逐步传递到整个序列，最终实现近似全局建模。
        # full_attention是标准Transformer的原始设计，每个token与序列中所有其他token计算注意力权重，能直接捕获任意两个位置间的依赖关系。
        self.sliding_window = None # config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor, # (b, s, hid 896)
        position_embeddings: tuple[torch.Tensor, torch.Tensor], # 形状都是 (1, s, head_dim 64)
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None, # LongTensor1D
        # **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1] # (b, s)
        hidden_shape = (*input_shape, -1, self.head_dim) # (b, s, -1, head_dim 64)

        # (b, s, hid/896) Linear (b, s, hid/896) view (b, s, 14, head_dim/64) transpose (b, 14, s, head_dim 64)
        query_states = self.q_proj.forward(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj.forward(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj.forward(hidden_states).view(hidden_shape).transpose(1, 2)

        # 旋转矩阵
        cos, sin = position_embeddings # 形状都是 (1, s, head_dim 64)
        # 输出维度不变，还是 (b, 14, s, head_dim 64)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # attention_interface: Callable = eager_attention_forward
        # if self.config._attn_implementation != "eager":
        #     attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = self.eager_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            # sliding_window=self.sliding_window,  # main diff with Llama
            # **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj.forward(attn_output)
        return attn_output, attn_weights


    def eager_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        # **kwargs: Unpack[TransformersKwargs],
    ):
        key_states = Attention.repeat_kv(key, self.num_key_value_groups)
        value_states = Attention.repeat_kv(value, self.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        # attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights

    @staticmethod
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Cache,
    ):
        inputs_embeds = self.embed_tokens.forward(input_ids) # (b, s, hid)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange( # LongTensor1D
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0) # (1, s)

        hidden_states = inputs_embeds

        # position_embeddings = (cos, sin) 两个张量，形状都是 (1, s, head_dim/64)
        position_embeddings = self.rotary_emb.forward(hidden_states, position_ids)


