import math
import torch
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        factory_kwargs = {"device": device, "dtype": dtype}

        # PyTorch nn.Linear 的 weight 形状是:
        #   (out_features, in_features)
        #
        # 数学上 forward 是:
        #   y = x @ W.T
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, **factory_kwargs)
        )

        self.reset_parameters()

    def reset_parameters(self):
        # Assignment 推荐的 Linear 初始化：
        # std = sqrt(2 / (in_features + out_features))
        std = math.sqrt(2.0 / (self.in_features + self.out_features))

        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        # weight.T: (in_features, out_features)
        # output: (..., out_features)
        return x @ self.weight.T

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias=False"
