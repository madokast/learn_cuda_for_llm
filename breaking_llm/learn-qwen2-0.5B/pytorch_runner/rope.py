import torch
from torch import nn, Tensor
from functools import wraps
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .model import ModelConfig

def dynamic_rope_update(rope_forward):
    """
    Decorator function to update the RoPE parameters in the forward pass, if the model is using a dynamic RoPE
    (i.e. a RoPE implementation that may recompute its frequencies in the forward pass).

    Args:
        rope_forward (Callable):
            The forward pass of the RoPE implementation.

    Returns:
        The decorated forward pass.
    """

    def longrope_frequency_update(self, position_ids, device):
        """Longrope uses long factor if sequence is larger than original pretraining length, short otherwise."""
        seq_len = torch.max(position_ids) + 1
        if hasattr(self.config, "original_max_position_embeddings"):
            original_max_position_embeddings = self.config.original_max_position_embeddings
        else:
            original_max_position_embeddings = self.config.max_position_embeddings
        if seq_len > original_max_position_embeddings:
            if not hasattr(self, "long_inv_freq"):
                self.long_inv_freq, _ = self.rope_init_fn(
                    self.config, device, seq_len=original_max_position_embeddings + 1
                )
            self.register_buffer("inv_freq", self.long_inv_freq, persistent=False)
        else:
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)

    def dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @wraps(rope_forward)
    def wrapper(self, x, position_ids):
        if "dynamic" in self.rope_type:
            dynamic_frequency_update(self, position_ids, device=x.device)
        elif self.rope_type == "longrope":
            longrope_frequency_update(self, position_ids, device=x.device)
        return rope_forward(self, x, position_ids)

    return wrapper



class RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: 'ModelConfig', device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        # if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
        #     self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        # else:
        # self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings # 32768
        self.original_max_seq_len = config.max_position_embeddings # 32768

        # self.config = config
        # self.rope_init_fn = _compute_default_rope_parameters # ROPE_INIT_FUNCTIONS[self.rope_type]

        # inv_freq 意思是 inverse frequency 形状为 (32, )，即 head_dim/2=64/2
        # self.attention_scaling = 1.0
        inv_freq, self.attention_scaling = _compute_default_rope_parameters(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        """
        x 就是 hidden_states 形状 (b, s, hid/896)
        position_ids 形状 (1, s)
        返回 (cos, sin) 形状都是 (1, s, head_dim/64)
        """
        # inv_freq 形状 (32, )
        # inv_freq[None, :, None] 则形状 (1, 32, 1)，相当于 .unsqueeze(0).unsqueeze(-1)
        # expand(1,-1,1) 后还是 (1, 32, 1)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)

        # position_ids 形状 (1, s)
        # position_ids[:, None, :] 则形状 (1, 1, s)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"

        # 显式地禁用自动混合精度
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32

            # inv_freq_expanded (1, 32, 1) @ position_ids_expanded (1, 1, s) = (1, 32, s)
            # 然后 transpose(1, 2) 得到 dao (1, s, 32)
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)

            # emb (1, s, 64)
            emb = torch.cat((freqs, freqs), dim=-1)

            # cos (1, s, 64) 
            # sin (1, s, 64)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling


        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _compute_default_rope_parameters(
    config: 'ModelConfig',
    device: Optional["torch.device"] = None,
    # seq_len: Optional[int] = None,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    base = config.rope_theta # 1000000.0
    partial_rotary_factor = 1.0 # config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads # 896 / 14 = 64
    dim = int(head_dim * partial_rotary_factor) # 64

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    # inv_freq 的形状是 (32, )
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor

def apply_rotary_pos_emb(q:Tensor, k:Tensor, cos:Tensor, sin:Tensor, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor. # 形状 (b, 14, s, head_dim 64)
        k (`torch.Tensor`): The key tensor.   # 形状 (b, 14, s, head_dim 64)
        cos (`torch.Tensor`): The cosine part of the rotary embedding. # 形状 (1, s, head_dim 64)
        sin (`torch.Tensor`): The sine part of the rotary embedding.   # 形状 (1, s, head_dim 64)
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # unsqueeze 用来在指定位置增加一个维度
    # 变成 (1, 1, s, head_dim 64)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # 括号内都是 (b, 14, s, head_dim 64) * (1, 1, s, head_dim 64)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    # 返回形状 (b, 14, s, head_dim 64)
    return q_embed, k_embed

def rotate_half(x:Tensor):
    """Rotates half the hidden dims of the input."""
    # x 是 qk 矩阵，形状 (b, 14, s, head_dim 64)

    # x1 是 qk 的前半部分，形状 (b, 14, s, 32)
    # x2 是 qk 的后半部分，形状 (b, 14, s, 32)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    # 返回 形状 (b, 14, s, head_dim 64)
    return torch.cat((-x2, x1), dim=-1)

