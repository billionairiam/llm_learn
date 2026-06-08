"""Batch size sweep: trains with different batch sizes (adjusting steps to keep total tokens constant).

Strategy:
  - Fix total tokens processed (default: ~40M, same as the LR sweep baseline).
  - For each batch_size, compute max_iters = total_tokens / (batch_size * context_length).
  - Optionally pair each batch size with a specific LR via --bs-lr-pairs; otherwise use --lr-max.
  - Cosine schedule decays at exactly max_iters per run.
  - Plots loss vs gradient step AND loss vs wall-clock time (the latter is the fairer comparison
    since larger batches do fewer steps but each step is slower).

Usage:
  cd assignment1-basics

  # Default batch sizes, same LR for all
  uv run python cs336_basics/bs_sweep.py \\
      --train-data data/tinystories_train.npy \\
      --val-data   data/tinystories_val.npy \\
      --device cpu --lr-max 1.58e-03

  # Explicit batch sizes
  uv run python cs336_basics/bs_sweep.py \\
      --train-data data/tinystories_train.npy \\
      --val-data   data/tinystories_val.npy \\
      --bs-list 1 4 16 32 64 128

  # Pair each batch size with a tuned LR (format: bs:lr)
  uv run python cs336_basics/bs_sweep.py \\
      --train-data data/tinystories_train.npy \\
      --val-data   data/tinystories_val.npy \\
      --bs-lr-pairs 8:5e-4 32:1.58e-03 64:3e-3 128:6e-3
"""

from __future__ import annotations

import argparse
import copy
import json
from argparse import Namespace
from pathlib import Path

from cs336_basics.train import train


def build_train_args(
    base_args: Namespace,
    batch_size: int,
    lr_max: float,
    max_iters: int,
) -> Namespace:
    """Clone base_args and override batch-size / LR / iters for one sweep run."""
    args = copy.deepcopy(base_args)
    args.batch_size = batch_size
    args.lr_max = lr_max
    args.lr_min = lr_max * 0.1
    args.max_iters = max_iters
    args.cosine_cycle_iters = max_iters
    args.warmup_iters = max(1, max_iters // 10)
    args.eval_every = max(1, max_iters // 10)
    args.log_every = max(1, max_iters // 100)
    args.run_name = f"bs_sweep_bs{batch_size}_lr{lr_max:.2e}"
    args.log_dir = str(Path(base_args.output_dir) / "logs")
    args.checkpoint_dir = str(Path(base_args.output_dir) / f"ckpt_bs{batch_size}")
    args.use_wandb = False
    return args


def extract_results(train_result: dict, batch_size: int, lr_max: float, total_tokens: int) -> dict:
    """Extract plottable data from train() return value."""
    records = train_result["records"]

    train_records = []
    val_records = []
    for r in records:
        m = r.metrics
        entry = {"step": r.step, "wall_time": r.wall_time}
        if "train_loss" in m:
            entry["train_loss"] = m["train_loss"]
            train_records.append(entry)
        if "val_loss" in m:
            val_records.append({**entry, "val_loss": m["val_loss"]})

    # Compute tokens_seen for each train record
    for rec in train_records:
        rec["tokens_seen"] = rec["step"] * batch_size * 256  # context_length filled below

    final_val_loss = train_result.get("final_val_loss")
    if final_val_loss is None and val_records:
        final_val_loss = val_records[-1]["val_loss"]
    final_train_loss = train_records[-1]["train_loss"] if train_records else None

    return {
        "batch_size": batch_size,
        "lr_max": lr_max,
        "total_tokens": total_tokens,
        "diverged": train_result["diverged"],
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "train_records": train_records,
        "val_records": val_records,
    }


def plot_results(all_results: list[dict], output_dir: Path, context_length: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots. Install with: uv add matplotlib")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Loss vs Step ---
    ax = axes[0]
    for r in all_results:
        records = r["train_records"]
        if not records:
            continue
        label = f"bs={r['batch_size']} lr={r['lr_max']:.1e}"
        if r["diverged"]:
            label += " (div)"
        ax.plot([rec["step"] for rec in records], [rec["train_loss"] for rec in records], label=label, alpha=0.8)
    ax.set_xlabel("Gradient Step")
    ax.set_ylabel("Train Loss")
    ax.set_title("Train Loss vs. Step")
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=0, top=12)
    ax.grid(True, alpha=0.3)

    # --- Loss vs Wall-clock Time (fair comparison) ---
    ax = axes[1]
    for r in all_results:
        records = r["train_records"]
        if not records:
            continue
        label = f"bs={r['batch_size']} lr={r['lr_max']:.1e}"
        if r["diverged"]:
            label += " (div)"
        ax.plot([rec["wall_time"] for rec in records], [rec["train_loss"] for rec in records], label=label, alpha=0.8)
    ax.set_xlabel("Wall-clock Time (s)")
    ax.set_ylabel("Train Loss")
    ax.set_title("Train Loss vs. Wall-clock Time")
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=0, top=12)
    ax.grid(True, alpha=0.3)

    # --- Loss vs Tokens Seen ---
    ax = axes[2]
    for r in all_results:
        records = r["train_records"]
        if not records:
            continue
        label = f"bs={r['batch_size']} lr={r['lr_max']:.1e}"
        if r["diverged"]:
            label += " (div)"
        tokens = [rec["step"] * r["batch_size"] * context_length for rec in records]
        ax.plot(tokens, [rec["train_loss"] for rec in records], label=label, alpha=0.8)
    ax.set_xlabel("Tokens Seen")
    ax.set_ylabel("Train Loss")
    ax.set_title("Train Loss vs. Tokens Seen")
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=0, top=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "bs_sweep.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch size sweep for TinyStories training")

    parser.add_argument("--config", type=str, default="../config/config.json")
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="bs_sweep_results")

    # Batch size list
    parser.add_argument("--bs-list", type=int, nargs="+", default=[1, 8, 32, 64, 128],
                        help="Batch sizes to sweep")
    # Optional per-BS LR (format: bs:lr, e.g. 32:1e-3 64:2e-3)
    parser.add_argument("--bs-lr-pairs", type=str, nargs="+", default=None,
                        help="Per-batch-size LR overrides, format bs:lr (e.g. 32:1e-3)")
    parser.add_argument("--lr-max", type=float, default=1e-3, help="Base LR (used with --base-bs for scaling)")

    # LR scaling: scale LR linearly or sqrt with batch size relative to a base
    parser.add_argument("--lr-scaling", type=str, default="linear", choices=["none", "linear", "sqrt"],
                        help="How to scale LR with batch size: none (fixed), linear, sqrt")
    parser.add_argument("--base-bs", type=int, default=32,
                        help="Base batch size for LR scaling (lr-max is the LR at this batch size)")

    # Total tokens budget (steps auto-computed per batch size)
    parser.add_argument("--max-iters", type=int, default=5000,
                        help="Fixed number of gradient steps per run (all batch sizes use the same)")

    # Training
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=100000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Optimizer
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)

    # System
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--compile", action="store_true")

    sweep_args = parser.parse_args()

    # Load model config
    config_path = Path(sweep_args.config)
    with open(config_path) as f:
        cfg = json.load(f)
    context_length = cfg["context_length"]

    # Parse per-BS LR mapping
    bs_lr_map: dict[int, float] = {}
    if sweep_args.bs_lr_pairs:
        for pair in sweep_args.bs_lr_pairs:
            bs_str, lr_str = pair.split(":")
            bs_lr_map[int(bs_str)] = float(lr_str)

    batch_sizes = sorted(sweep_args.bs_list)

    def compute_lr(bs: int) -> float:
        """Compute scaled LR for a given batch size."""
        if bs in bs_lr_map:
            return bs_lr_map[bs]
        ratio = bs / sweep_args.base_bs
        if sweep_args.lr_scaling == "linear":
            return sweep_args.lr_max * ratio
        elif sweep_args.lr_scaling == "sqrt":
            return sweep_args.lr_max * (ratio ** 0.5)
        else:  # none
            return sweep_args.lr_max

    print(f"Model config: {cfg}")
    print(f"Fixed steps per run: {sweep_args.max_iters}")
    print(f"LR scaling: {sweep_args.lr_scaling} (base: bs={sweep_args.base_bs}, lr={sweep_args.lr_max:.2e})")
    print(f"Batch sizes: {batch_sizes}")
    print()

    for bs in batch_sizes:
        lr = compute_lr(bs)
        total_tokens = sweep_args.max_iters * bs * context_length
        print(f"  bs={bs:>4d}  →  steps={sweep_args.max_iters:>6,}  lr={lr:.2e}  tokens={total_tokens:>12,}")

    output_dir = Path(sweep_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build base args
    base_args = Namespace(
        config=sweep_args.config,
        train_data=sweep_args.train_data,
        val_data=sweep_args.val_data,
        checkpoint_dir="checkpoints",
        resume_from=None,
        vocab_size=cfg.get("vocab_size"),
        context_length=context_length,
        d_model=cfg.get("d_model"),
        num_layers=cfg.get("num_layers"),
        num_heads=cfg.get("num_heads"),
        d_ff=cfg.get("d_ff"),
        rope_theta=cfg.get("rope_theta"),
        weight_decay=sweep_args.weight_decay,
        beta1=sweep_args.beta1,
        beta2=sweep_args.beta2,
        eps=sweep_args.eps,
        max_grad_norm=sweep_args.max_grad_norm,
        eval_iters=sweep_args.eval_iters,
        save_every=sweep_args.save_every,
        device=sweep_args.device,
        seed=sweep_args.seed,
        compile=sweep_args.compile,
        use_wandb=False,
        wandb_project="cs336",
        run_name=None,
        log_dir=str(output_dir / "logs"),
        output_dir=str(output_dir),
        # Placeholders overridden per run
        batch_size=None,
        lr_max=None,
        lr_min=None,
        max_iters=None,
        warmup_iters=None,
        cosine_cycle_iters=None,
        log_every=None,
        eval_every=None,
    )

    # Run sweep
    all_results: list[dict] = []

    for i, bs in enumerate(batch_sizes):
        lr = compute_lr(bs)
        max_iters = sweep_args.max_iters
        total_tokens = max_iters * bs * context_length

        print(f"\n{'='*60}")
        print(f"Run {i+1}/{len(batch_sizes)}:  batch_size={bs}  lr_max={lr:.2e}  "
              f"steps={max_iters}  tokens={total_tokens:,}")
        print(f"{'='*60}")

        run_args = build_train_args(base_args, batch_size=bs, lr_max=lr, max_iters=max_iters)
        train_result = train(run_args)
        result = extract_results(train_result, batch_size=bs, lr_max=lr, total_tokens=total_tokens)
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("BATCH SIZE SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"{'BS':>6s}  {'LR':>10s}  {'Tokens':>14s}  {'Train Loss':>12s}  {'Val Loss':>12s}  {'Status':>10s}")
    print("-" * 75)

    for r in all_results:
        status = "DIVERGED" if r["diverged"] else "OK"
        tl = f"{r['final_train_loss']:.4f}" if r["final_train_loss"] is not None else "N/A"
        vl = f"{r['final_val_loss']:.4f}" if r["final_val_loss"] is not None else "N/A"
        tokens = r["total_tokens"]
        print(f"{r['batch_size']:>6d}  {r['lr_max']:>10.2e}  {tokens:>14,}  {tl:>12s}  {vl:>12s}  {status:>10s}")

    # Save results
    json_path = output_dir / "bs_sweep_results.json"
    json_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {json_path}")

    # Plot
    plot_results(all_results, output_dir, context_length)


if __name__ == "__main__":
    main()
