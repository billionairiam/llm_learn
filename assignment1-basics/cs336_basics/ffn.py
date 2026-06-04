import torch
import math
from torch import nn

# x.shape: (batch_size, sequence_length, d_model)
# W1 W3 (d_ff, d_model)
# W2 (d_model d_model)
class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.d_model = d_model
        if d_ff == None:
            d_ff = math.ceil((8 * d_model / 3) / 64) * 64

        self.d_ff = d_ff
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))

        self.reset_parameter()

    def reset_parameter(self):
        std1 = math.sqrt( 2 / (self.d_ff + self.d_model))
        std3 = std1
        std2 = math.sqrt( 2 / (self.d_model + self.d_model))

        nn.init.trunc_normal_(self.w1, mean=0.0, std=std1, a=-3 * std1, b=3 * std1)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=std2, a=-3 * std2, b=3 * std2)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=std3, a=-3 * std3, b=3 * std3)

    def forward(self, x: torch.Tensor):
        x_w1 = torch.einsum("... d, h d -> ... h", x, self.w1)
        x_w3 = torch.einsum("... d, h d -> ... h", x, self.w3)

        silu = x_w1 * torch.sigmoid(x_w1)
        hidden = silu * x_w3

        out = torch.einsum("... h, d h -> ... d", hidden, self.w2)
        return out
