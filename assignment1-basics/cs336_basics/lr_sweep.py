"""Learning rate sweep: runs train.train() with different LR values, then plots comparison curves.

Strategy:
  - Log-spaced LR sweep from small to large (includes at least one divergent run).
  - Each run reuses train() from train.py — same model init, data loading, checkpointing, tracking.
  - Cosine schedule decays to min_lr at exactly max_iters (as recommended by the assignment).
  - Divergence detected by train() (NaN/Inf or loss > 100).
  - Outputs: JSON results + loss-vs-step / loss-vs-walltime plots.

Usage:
  cd assignment1-basics
  uv run python cs336_basics/lr_sweep.py \
      --train-data data/tinystories_train.npy \
      --val-data   data/tinystories_val.npy \
      --device cpu
"""

from __future__ import annotations

import argparse
import copy
import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from cs336_basics.train import train


def build_train_args(
    base_args: Namespace,
    lr_max: float,
    run_index: int,
) -> Namespace:
    """Clone base_args and override LR-related fields for one sweep run."""
    args = copy.deepcopy(base_args)
    args.lr_max = lr_max
    args.lr_min = lr_max * 0.1
    # Cosine decay ends exactly at max_iters
    args.cosine_cycle_iters = args.max_iters
    # Per-run naming
    args.run_name = f"lr_sweep_{lr_max:.2e}"
    args.log_dir = str(Path(base_args.output_dir) / "logs")
    args.checkpoint_dir = str(Path(base_args.output_dir) / f"ckpt_lr{lr_max:.2e}")
    # No wandb for sweep (too noisy); use JSON logs
    args.use_wandb = False
    return args


def plot_results(all_results: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots. Install with: uv add matplotlib")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss vs Step ---
    ax = axes[0]
    for r in all_results:
        records = r["train_records"]
        if not records:
            continue
        steps = [rec["step"] for rec in records]
        losses = [rec["train_loss"] for rec in records]
        label = f"lr={r['lr_max']:.1e}"
        if r["diverged"]:
            label += " (diverged)"
        ax.plot(steps, losses, label=label, alpha=0.8)
    ax.set_xlabel("Gradient Step")
    ax.set_ylabel("Train Loss")
    ax.set_title("Train Loss vs. Step")
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=0, top=15)
    ax.grid(True, alpha=0.3)

    # --- Loss vs Wall-clock Time ---
    ax = axes[1]
    for r in all_results:
        records = r["train_records"]
        if not records:
            continue
        times = [rec["wall_time"] for rec in records]
        losses = [rec["train_loss"] for rec in records]
        label = f"lr={r['lr_max']:.1e}"
        if r["diverged"]:
            label += " (diverged)"
        ax.plot(times, losses, label=label, alpha=0.8)
    ax.set_xlabel("Wall-clock Time (s)")
    ax.set_ylabel("Train Loss")
    ax.set_title("Train Loss vs. Wall-clock Time")
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=0, top=15)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "lr_sweep.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")

    # --- Summary bar chart: final val loss per LR ---
    converged = [r for r in all_results if not r["diverged"] and r.get("final_val_loss") is not None]
    if converged:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        lrs = [f"{r['lr_max']:.1e}" for r in converged]
        val_losses = [r["final_val_loss"] for r in converged]
        colors = ["#2ecc71" if v <= 2.0 else "#e74c3c" for v in val_losses]
        ax2.bar(lrs, val_losses, color=colors)
        ax2.set_xlabel("Max Learning Rate")
        ax2.set_ylabel("Final Val Loss")
        ax2.set_title("Final Validation Loss per Learning Rate")
        ax2.axhline(y=2.0, color="gray", linestyle="--", label="target ≤ 2.0 (low-resource)")
        ax2.axhline(y=1.45, color="orange", linestyle="--", label="target ≤ 1.45 (full)")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        bar_path = output_dir / "lr_sweep_val_summary.png"
        fig2.savefig(bar_path, dpi=150)
        plt.close(fig2)
        print(f"Summary plot saved to {bar_path}")


def extract_results(train_result: dict, lr_max: float) -> dict:
    """Extract plottable data from train() return value."""
    records = train_result["records"]

    train_records = []
    val_records = []
    for r in records:
        m = r.metrics
        entry = {"step": r.step, "wall_time": r.wall_time}
        if "train_loss" in m:
            entry["train_loss"] = m["train_loss"]
            entry["lr"] = m.get("lr")
            train_records.append(entry)
        if "val_loss" in m:
            val_records.append({**entry, "val_loss": m["val_loss"], "val_ppl": m.get("val_ppl")})

    # Prefer the dedicated final_val_loss from train(), fall back to last val record
    final_val_loss = train_result.get("final_val_loss")
    if final_val_loss is None and val_records:
        final_val_loss = val_records[-1]["val_loss"]
    final_train_loss = train_records[-1]["train_loss"] if train_records else None

    return {
        "lr_max": lr_max,
        "diverged": train_result["diverged"],
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "train_records": train_records,
        "val_records": val_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Learning rate sweep for TinyStories training")

    parser.add_argument("--config", type=str, default="../config/config.json")
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="lr_sweep_results")

    # LR sweep range
    parser.add_argument("--lr-list", type=float, nargs="+", default=None,
                        help="Explicit list of LRs to sweep (overrides --lr-min/max-sweep)")
    parser.add_argument("--lr-min-sweep", type=float, default=1e-4, help="Smallest max-LR to try")
    parser.add_argument("--lr-max-sweep", type=float, default=1e-2, help="Largest max-LR to try")
    parser.add_argument("--num-lrs", type=int, default=6, help="Number of LR values to sweep")

    # Training (passed through to train.py)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--warmup-iters", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=100000)  # effectively disabled for sweep
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Optimizer
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)

    # System
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--compile", action="store_true", help="torch.compile the model")

    sweep_args = parser.parse_args()

    # Determine LR list
    if sweep_args.lr_list is not None:
        lr_values = sorted(sweep_args.lr_list)
    else:
        lr_values = np.logspace(
            np.log10(sweep_args.lr_min_sweep), np.log10(sweep_args.lr_max_sweep), sweep_args.num_lrs
        ).tolist()

    # Load model config to show total tokens
    config_path = Path(sweep_args.config)
    with open(config_path) as f:
        cfg = json.load(f)

    total_tokens = sweep_args.max_iters * sweep_args.batch_size * cfg["context_length"]
    print(f"Model config: {cfg}")
    print(f"Sweeping {len(lr_values)} learning rates: {[f'{v:.1e}' for v in lr_values]}")
    print(f"Each run: {sweep_args.max_iters} steps × bs={sweep_args.batch_size} × ctx={cfg['context_length']}")
    print(f"Total tokens per run: {total_tokens:,}")

    output_dir = Path(sweep_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a base Namespace compatible with train()
    base_args = Namespace(
        config=sweep_args.config,
        train_data=sweep_args.train_data,
        val_data=sweep_args.val_data,
        checkpoint_dir="checkpoints",
        resume_from=None,
        # Model params filled from config by setting to None
        vocab_size=None,
        context_length=None,
        d_model=None,
        num_layers=None,
        num_heads=None,
        d_ff=None,
        rope_theta=None,
        # Optimizer (will be overridden per-run for lr_max/lr_min)
        lr_max=None,
        lr_min=None,
        weight_decay=sweep_args.weight_decay,
        beta1=sweep_args.beta1,
        beta2=sweep_args.beta2,
        eps=sweep_args.eps,
        max_grad_norm=sweep_args.max_grad_norm,
        # Schedule
        warmup_iters=sweep_args.warmup_iters,
        cosine_cycle_iters=sweep_args.max_iters,
        # Training
        batch_size=sweep_args.batch_size,
        max_iters=sweep_args.max_iters,
        eval_iters=sweep_args.eval_iters,
        log_every=sweep_args.log_every,
        eval_every=sweep_args.eval_every,
        save_every=sweep_args.save_every,
        # System
        device=sweep_args.device,
        seed=sweep_args.seed,
        compile=sweep_args.compile,
        # Tracking
        use_wandb=False,
        wandb_project="cs336",
        run_name=None,
        log_dir=str(output_dir / "logs"),
        output_dir=str(output_dir),
    )

    # Fill model config from JSON (same logic as train.parse_args)
    for key, value in cfg.items():
        attr = key.replace("-", "_")
        if hasattr(base_args, attr) and getattr(base_args, attr) is None:
            setattr(base_args, attr, value)

    # Run sweep
    all_results: list[dict] = []

    for i, lr in enumerate(lr_values):
        print(f"\n{'='*60}")
        print(f"Run {i+1}/{len(lr_values)}:  lr_max = {lr:.2e}")
        print(f"{'='*60}")

        run_args = build_train_args(base_args, lr_max=lr, run_index=i)
        train_result = train(run_args)
        result = extract_results(train_result, lr_max=lr)
        all_results.append(result)

    # Print summary table
    print(f"\n{'='*60}")
    print("SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"{'LR':>12s}  {'Train Loss':>12s}  {'Val Loss':>12s}  {'Status':>12s}")
    print("-" * 55)

    best_result = None
    for r in all_results:
        status = "DIVERGED" if r["diverged"] else "OK"
        tl = f"{r['final_train_loss']:.4f}" if r["final_train_loss"] is not None else "N/A"
        vl = f"{r['final_val_loss']:.4f}" if r["final_val_loss"] is not None else "N/A"
        print(f"{r['lr_max']:>12.2e}  {tl:>12s}  {vl:>12s}  {status:>12s}")

        if not r["diverged"] and r.get("final_val_loss") is not None:
            if best_result is None or r["final_val_loss"] < best_result["final_val_loss"]:
                best_result = r

    if best_result:
        print(f"\nBest LR: {best_result['lr_max']:.2e}  (val_loss={best_result['final_val_loss']:.4f})")

    # Edge of stability analysis
    divergent = [r for r in all_results if r["diverged"]]
    convergent = [r for r in all_results if not r["diverged"]]
    if divergent and convergent:
        edge_lr = max(r["lr_max"] for r in convergent)
        first_diverge = min(r["lr_max"] for r in divergent)
        print(f"Edge of stability: largest converging LR = {edge_lr:.2e}, "
              f"smallest diverging LR = {first_diverge:.2e}")
        if best_result:
            print(f"Best LR / divergence boundary = {best_result['lr_max'] / first_diverge:.2f}")

    # Save results (convert for JSON serialization)
    json_path = output_dir / "lr_sweep_results.json"
    json_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {json_path}")

    # Plot
    plot_results(all_results, output_dir)


if __name__ == "__main__":
    main()
