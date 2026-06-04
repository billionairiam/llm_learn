from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ====== 按你的项目实际路径修改这里 ======
from cs336_basics.data_loading import get_batch
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint

# 下面这几个名字按你自己的实现改
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.learning_rate_schedule import get_lr_cosine_schedule
from cs336_basics.gradient_clipping import gradient_clipping
# ======================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    """
    如果你自己的 TransformerLM 构造函数参数名不同，就只改这里。
    """
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
    )
    return model


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int,
) -> float:
    model.eval()

    losses: list[float] = []

    for _ in range(eval_iters):
        x, y = get_batch(
            x=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = args.device

    train_data = np.load(args.train_data, mmap_mode="r")

    val_data = None
    if args.val_data is not None:
        val_data = np.load(args.val_data, mmap_mode="r")

    model = build_model(args).to(device)

    # def check_model_parameters(model: torch.nn.Module) -> None:
    #     bad = False

    #     for name, param in model.named_parameters():
    #         finite = torch.isfinite(param)

    #         if not finite.all():
    #             bad = True
    #             num_bad = (~finite).sum().item()
    #             total = param.numel()

    #             print("=" * 80, flush=True)
    #             print(f"Parameter {name} has NaN/Inf", flush=True)
    #             print(f"shape: {tuple(param.shape)}", flush=True)
    #             print(f"bad: {num_bad}/{total}", flush=True)
    #             print(f"has nan: {torch.isnan(param).any().item()}", flush=True)
    #             print(f"has inf: {torch.isinf(param).any().item()}", flush=True)
    #             print(f"min finite: {param[finite].min().item() if finite.any() else 'no finite'}", flush=True)
    #             print(f"max finite: {param[finite].max().item() if finite.any() else 'no finite'}", flush=True)

    #     if bad:
    #         raise RuntimeError("Bad initial parameter")


    # def add_nan_hooks(model: torch.nn.Module) -> None:
    #     def hook(module, inputs, output):
    #         tensors = []

    #         if torch.is_tensor(output):
    #             tensors.append(output)
    #         elif isinstance(output, tuple):
    #             tensors.extend([x for x in output if torch.is_tensor(x)])

    #         for t in tensors:
    #             if not torch.isfinite(t).all():
    #                 print("=" * 80)
    #                 print("First module producing NaN/Inf:")
    #                 print(module)
    #                 print("output shape:", tuple(t.shape))
    #                 print("has nan:", torch.isnan(t).any().item())
    #                 print("has inf:", torch.isinf(t).any().item())
    #                 print("min:", torch.nan_to_num(t).min().item())
    #                 print("max:", torch.nan_to_num(t).max().item())
    #                 raise RuntimeError("NaN/Inf detected in forward")

    #     for module in model.modules():
    #         module.register_forward_hook(hook)


    # check_model_parameters(model)
    # add_nan_hooks(model)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    start_iter = 0

    if args.resume_from is not None:
        start_iter = load_checkpoint(
            src=args.resume_from,
            model=model,
            optimizer=optimizer,
        )
        print(f"Resumed from checkpoint {args.resume_from}, iteration={start_iter}")

    model.train()

    for iteration in range(start_iter, args.max_iters):
        lr = get_lr_cosine_schedule(
            it=iteration,
            max_learning_rate=args.lr_max,
            min_learning_rate=args.lr_min,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_batch(
            x=train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device,
        )

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if args.max_grad_norm is not None:
            gradient_clipping(
                parameters=model.parameters(),
                max_l2_norm=args.max_grad_norm,
            )

        optimizer.step()

        if iteration % args.log_every == 0:
            print(
                f"iter={iteration} "
                f"train_loss={loss.item():.4f} "
                f"lr={lr:.6e}"
            )

        if val_data is not None and iteration % args.eval_every == 0 and iteration > 0:
            val_loss = estimate_loss(
                model=model,
                dataset=val_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=device,
                eval_iters=args.eval_iters,
            )

            print(
                f"iter={iteration} "
                f"val_loss={val_loss:.4f} "
                f"val_ppl={math.exp(val_loss):.4f}"
            )

        if iteration % args.save_every == 0 and iteration > 0:
            ckpt_path = Path(args.checkpoint_dir) / f"ckpt_{iteration}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=iteration,
                out=ckpt_path,
            )

            print(f"Saved checkpoint to {ckpt_path}")

    final_path = Path(args.checkpoint_dir) / "ckpt_final.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=args.max_iters,
        out=final_path,
    )

    print(f"Saved final checkpoint to {final_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, default=None)

    # checkpoint
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume-from", type=str, default=None)

    # model
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    # optimizer
    parser.add_argument("--lr-max", type=float, default=3e-4)
    parser.add_argument("--lr-min", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # schedule
    parser.add_argument("--warmup-iters", type=int, default=1000)
    parser.add_argument("--cosine-cycle-iters", type=int, default=10000)

    # training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=10000)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1000)

    # system
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1337)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
