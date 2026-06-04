from collections.abc import Iterable
import torch


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    """
    Clip gradients in-place so that the global L2 norm of all gradients
    is at most max_l2_norm.
    """

    params = [p for p in parameters if p.grad is not None]

    if len(params) == 0:
        return

    with torch.no_grad():
        total_norm_sq = torch.zeros((), device=params[0].grad.device)

        for p in params:
            grad = p.grad
            total_norm_sq += torch.sum(grad * grad)

        total_norm = torch.sqrt(total_norm_sq)

        if total_norm > max_l2_norm:
            scale = max_l2_norm / (total_norm + eps)

            for p in params:
                p.grad.mul_(scale)
