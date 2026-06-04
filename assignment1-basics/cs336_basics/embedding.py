import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        factory_kwargs = {"device": device, "dtype": dtype}

        # shape: (vocab_size, d_model)
        # 每一行是一个 token 的 embedding vector
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, **factory_kwargs)
        )

        self.reset_parameters()

    def reset_parameters(self):
        # Assignment 推荐 Embedding 初始化：
        # N(0, 1)，并截断到 [-3, 3]
        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (...)
        # output: (..., embedding_dim)
        return self.weight[token_ids]

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"
