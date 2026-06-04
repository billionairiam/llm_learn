from collections.abc import Callable
from typing import Optional
import math
import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        beta1, beta2 = betas
        if not 0 <= beta1 < 1:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0 <= beta2 < 1:
            raise ValueError(f"Invalid beta2: {beta2}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    grad = p.grad
                    state = self.state[p]

                    if len(state) == 0:
                        state["t"] = 0
                        state["m"] = torch.zeros_like(p)
                        state["v"] = torch.zeros_like(p)

                    m = state["m"]
                    v = state["v"]

                    state["t"] += 1
                    t = state["t"]

                    # m_t = beta1 * m_{t-1} + (1 - beta1) * grad
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)

                    # v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # bias-corrected learning rate
                    lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                    # Adam update
                    p.addcdiv_(m, v.sqrt().add(eps), value=-lr_t)

                    # decoupled weight decay
                    if weight_decay != 0:
                        p.add_(p, alpha=-lr * weight_decay)

        return loss
