import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    # x.shape: (batch_size, sequence_length, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype

        x_fp32 = x.to(torch.float32)

        rms_inv = torch.rsqrt(
            torch.mean(x_fp32 * x_fp32, dim=-1, keepdim=True) + self.eps
        )

        out = x_fp32 * rms_inv
        out = out * self.weight

        return out.to(in_dtype)
