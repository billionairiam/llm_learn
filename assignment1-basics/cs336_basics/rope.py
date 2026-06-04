import torch
from torch import nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even for RoPE, got d_k={d_k}")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # inv_freq.shape: (d_k // 2,)
        #
        # 对应公式:
        #   1 / theta^(2k / d_k)
        #
        # 这里 torch.arange(0, d_k, 2) 得到:
        #   0, 2, 4, ..., d_k - 2
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float64) / d_k)
        )

        # positions.shape: (max_seq_len,)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float64)

        # angles.shape: (max_seq_len, d_k // 2)
        angles = torch.outer(positions, inv_freq)

        # cos/sin shape: (max_seq_len, d_k // 2)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        # x.shape: (..., seq_len, d_k)
        # token_positions.shape: (..., seq_len)

        if x.shape[-1] != self.d_k:
            raise ValueError(f"Expected x.shape[-1] == {self.d_k}, got {x.shape[-1]}")

        token_positions = token_positions.to(device=x.device, dtype=torch.long)

        # cos/sin shape: (..., seq_len, d_k // 2)
        cos = self.cos.to(device=x.device, dtype=x.dtype)[token_positions]
        sin = self.sin.to(device=x.device, dtype=x.dtype)[token_positions]

        # x_even/x_odd shape: (..., seq_len, d_k // 2)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # 每两个维度一组做 2D rotation
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        # stack 后 shape: (..., seq_len, d_k // 2, 2)
        # flatten 后 shape: (..., seq_len, d_k)
        out = torch.stack((out_even, out_odd), dim=-1).flatten(-2)

        return out
