import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits:  (..., vocab_size)
    targets: (...)
    return: scalar average cross-entropy loss
    """

    # 1. subtract max logit for numerical stability
    # max_logits shape: (..., 1)
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted_logits = logits - max_logits

    # 2. logsumexp over vocab dimension
    # shape: (...)
    log_sum_exp = torch.log(torch.exp(shifted_logits).sum(dim=-1))

    # 3. get the logit corresponding to the correct target token
    # targets shape: (...)
    # targets.unsqueeze(-1) shape: (..., 1)
    target_logits = torch.gather(
        shifted_logits,
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    # 4. cross entropy = logsumexp - target_logit
    loss = log_sum_exp - target_logits

    # 5. average over all batch-like dimensions
    return loss.mean()
