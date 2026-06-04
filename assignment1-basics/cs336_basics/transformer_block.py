import torch
from torch import nn, Tensor

from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.ffn import SwiGLU
from cs336_basics.multihead_self_attention import CausalMultiHeadSelfAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        use_rope: bool = True,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_rope = use_rope

        self.ln1 = RMSNorm(d_model=d_model, device=device)

        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
        )

        self.ln2 = RMSNorm(d_model=d_model, device=device)

        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
        )

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
    ) -> Tensor:
        """
        x: (..., sequence_length, d_model)

        return:
            (..., sequence_length, d_model)
        """

        # sub-layer 1: pre-norm MHA + residual
        x = x + self.attn(
            self.ln1(x),
            token_positions=token_positions,
        )

        # sub-layer 2: pre-norm FFN + residual
        x = x + self.ffn(self.ln2(x))

        return x
