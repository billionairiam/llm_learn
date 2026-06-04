import torch
from torch import nn, Tensor

from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    use_rope=True,
                    max_seq_len=context_length,
                    theta=theta,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(
            d_model=d_model,
            device=device,
        )

        # lm_head.weight shape: (vocab_size, d_model)
        # 不用 bias
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
        )

    def forward(
        self,
        input_ids: Tensor,
    ) -> Tensor:
        """
        input_ids: (..., sequence_length)

        return:
            logits: (..., sequence_length, vocab_size)
        """

        *batch_dims, seq_len = input_ids.shape

        if seq_len > self.context_length:
            raise ValueError(
                f"sequence length {seq_len} > context_length {self.context_length}"
            )

        x = self.token_embeddings(input_ids)

        token_positions = torch.arange(
            seq_len,
            device=input_ids.device,
            dtype=torch.long,
        )

        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        x = self.ln_final(x)

        # 等价于 self.lm_head(x)
        logits = torch.einsum(
            "... s d, v d -> ... s v",
            x,
            self.lm_head.weight,
        )

        return logits
