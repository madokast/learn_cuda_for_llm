from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelConfig:
    hidden_size: int = 896
    intermediate_size: int = 4864


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
