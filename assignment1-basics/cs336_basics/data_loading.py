import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    x: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # x[start : start + context_length]
    # x[start + 1 : start + context_length + 1]
    # 所以 start 最大只能到 len(x) - context_length - 1
    starts = np.random.randint(
        low=0,
        high=len(x) - context_length,
        size=batch_size,
    )

    input_batch = np.stack([
        x[start : start + context_length]
        for start in starts
    ])

    target_batch = np.stack([
        x[start + 1 : start + context_length + 1]
        for start in starts
    ])

    input_tensor = torch.from_numpy(input_batch).long().to(device)
    target_tensor = torch.from_numpy(target_batch).long().to(device)

    return input_tensor, target_tensor
