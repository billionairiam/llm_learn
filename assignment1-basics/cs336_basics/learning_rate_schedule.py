import math


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate

    if it > cosine_cycle_iters:
        return min_learning_rate

    return min_learning_rate + 0.5 * (
        1.0
        + math.cos(
            math.pi
            * (it - warmup_iters)
            / (cosine_cycle_iters - warmup_iters)
        )
    ) * (max_learning_rate - min_learning_rate)
