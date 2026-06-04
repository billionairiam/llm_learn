import math
import torch
from torch import nn, Tensor
from cs336_basics.rope import RotaryPositionalEmbedding

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # 1. numerical stability trick
    x_shifted = x - torch.max(x, dim=dim, keepdim=True).values

    # 2. exponentiate
    exp_x = torch.exp(x_shifted)

    # 3. normalize along dim
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    q: (..., seq_len_q, d_k)
    k: (..., seq_len_k, d_k)
    v: (..., seq_len_k, d_v)
    mask: (seq_len_q, seq_len_k), True 表示可以 attend，False 表示不能 attend

    return: (..., seq_len_q, d_v)
    """

    d_k = q.shape[-1]

    # (..., seq_len_q, d_k) @ (..., d_k, seq_len_k)
    # -> (..., seq_len_q, seq_len_k)
    scores = q @ k.transpose(-2, -1)

    # scaled
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        # False 的地方填成 -inf，softmax 后概率就是 0
        scores = scores.masked_fill(~mask, float("-inf"))

    # 对 key 那一维做 softmax
    attn_weights = softmax(scores, dim=-1)

    # (..., seq_len_q, seq_len_k) @ (..., seq_len_k, d_v)
    # -> (..., seq_len_q, d_v)
    output = attn_weights @ v

    return output


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model must be divisible by num_heads, got {d_model=} {num_heads=}"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope

        # qkv_proj_weight shape: (3 * d_model, d_model)
        # 对应 PyTorch Linear.weight 的习惯: (out_features, in_features)
        self.qkv_proj_weight = nn.Parameter(
            torch.empty(3 * d_model, d_model, device=device)
        )

        # o_proj_weight shape: (d_model, d_model)
        self.o_proj_weight = nn.Parameter(
            torch.empty(d_model, d_model, device=device)
        )

        if use_rope:
            if max_seq_len is None or theta is None:
                raise ValueError("use_rope=True 时必须传 max_seq_len 和 theta")

            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.head_dim,
                max_seq_len=max_seq_len,
                device=device,
            )
        else:
            self.rope = None
        
        self.reset_parameter()

    def reset_parameter(self):
        qkv_std = math.sqrt(2.0 / (self.d_model + 3 * self.d_model))
        out_std = math.sqrt(2.0 / (self.d_model + self.d_model))

        torch.nn.init.trunc_normal_(
            self.qkv_proj_weight,
            mean=0.0,
            std=qkv_std,
            a=-3 * qkv_std,
            b=3 * qkv_std,
        )

        torch.nn.init.trunc_normal_(
            self.o_proj_weight,
            mean=0.0,
            std=out_std,
            a=-3 * out_std,
            b=3 * out_std,
        )

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
        q_proj_weight: Tensor | None = None,
        k_proj_weight: Tensor | None = None,
        v_proj_weight: Tensor | None = None,
        o_proj_weight: Tensor | None = None,
    ) -> Tensor:
        """
        x: (..., sequence_length, d_model)

        return:
            (..., sequence_length, d_model)
        """

        *batch_dims, seq_len, d_model = x.shape

        if d_model != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {d_model}")

        # 如果 adapter 传了外部权重，就用外部权重
        if q_proj_weight is not None:
            if (
                k_proj_weight is None
                or v_proj_weight is None
                or o_proj_weight is None
            ):
                raise ValueError("q/k/v/o weights must be provided together")

            # q/k/v weight: (d_model, d_model)
            # 合并成一次 QKV projection
            qkv_weight = torch.cat(
                [q_proj_weight, k_proj_weight, v_proj_weight],
                dim=0,
            )
            out_weight = o_proj_weight
        else:
            qkv_weight = self.qkv_proj_weight
            out_weight = self.o_proj_weight

        # ------------------------------------------------------------
        # 1. 一次性算 Q/K/V
        #
        # x:          (..., seq_len, d_model)
        # qkv_weight: (3 * d_model, d_model)
        #
        # 等价于:
        #   qkv = x @ qkv_weight.T
        #
        # 输出:
        #   qkv: (..., seq_len, 3 * d_model)
        # ------------------------------------------------------------
        qkv = torch.einsum(
            "... s d, o d -> ... s o",
            x,
            qkv_weight,
        )

        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)

        # ------------------------------------------------------------
        # 2. split heads
        #
        # q/k/v:
        #   (..., seq_len, d_model)
        #
        # reshape 后:
        #   (..., seq_len, num_heads, head_dim)
        #
        # transpose 后:
        #   (..., num_heads, seq_len, head_dim)
        # ------------------------------------------------------------
        q = q.reshape(*batch_dims, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(*batch_dims, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(*batch_dims, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)

        # ------------------------------------------------------------
        # 3. optional RoPE
        #
        # RoPE 只作用在 q/k 上，不作用在 v 上
        # ------------------------------------------------------------
        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)

            else:
                token_positions = token_positions.to(
                    device=x.device,
                    dtype=torch.long,
                )

                # q/k shape:
                #   (..., num_heads, seq_len, head_dim)
                #
                # token_positions 原始 shape:
                #   (..., seq_len)
                #
                # 插入 head 维度，变成:
                #   (..., 1, seq_len)
                #
                # 这样可以 broadcast 到 num_heads
                token_positions = token_positions.unsqueeze(-2)

            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # ------------------------------------------------------------
        # 4. causal mask
        #
        # True  = 可以看
        # False = 不可以看
        # ------------------------------------------------------------
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )

        # 复用你已经写好的 scaled_dot_product_attention
        # out: (..., num_heads, seq_len, head_dim)
        out = scaled_dot_product_attention(q, k, v, causal_mask)

        # ------------------------------------------------------------
        # 5. concat heads
        #
        # (..., num_heads, seq_len, head_dim)
        # -> (..., seq_len, num_heads, head_dim)
        # -> (..., seq_len, d_model)
        # ------------------------------------------------------------
        out = out.transpose(-3, -2)
        out = out.reshape(*batch_dims, seq_len, self.d_model)

        # ------------------------------------------------------------
        # 6. output projection
        #
        # 等价于:
        #   out = out @ out_weight.T
        # ------------------------------------------------------------
        out = torch.einsum(
            "... s d, o d -> ... s o",
            out,
            out_weight,
        )

        return out
