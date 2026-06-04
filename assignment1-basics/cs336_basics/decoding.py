#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# 根据你自己的项目路径修改这里
# 假设你的模型类叫 TransformerLM
# 假设你的 tokenizer 类叫 Tokenizer
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.tokenizer import Tokenizer


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """
    兼容几种常见 checkpoint 保存格式：

    1. torch.save(model.state_dict(), path)
    2. torch.save({"model_state_dict": model.state_dict(), ...}, path)
    3. torch.save({"model": model.state_dict(), ...}, path)
    4. torch.save({"state_dict": model.state_dict(), ...}, path)
    """
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # 可能 checkpoint 本身就是 state_dict
            state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format.")

    # 处理 DDP / torch.compile 可能带来的 module. 前缀
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        cleaned[k] = v

    return cleaned


def top_p_filter_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    logits: [batch_size, vocab_size]

    top-p sampling:
    只保留概率最高的一批 token，使它们的累计概率 >= top_p。
    其他 token 的 logits 设为 -inf。
    """
    if top_p is None or top_p >= 1.0:
        return logits

    if top_p <= 0.0:
        raise ValueError("top_p must be > 0.")

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # cumulative_probs > top_p 的位置要删除
    sorted_remove_mask = cumulative_probs > top_p

    # 关键：保留第一个使累计概率超过 top_p 的 token
    # 例如 p=0.9，累计到第 k 个才超过 0.9，那么第 k 个也要保留
    sorted_remove_mask[..., 1:] = sorted_remove_mask[..., :-1].clone()
    sorted_remove_mask[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(sorted_remove_mask, float("-inf"))

    filtered_logits = torch.empty_like(logits)
    filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    return filtered_logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    logits: [batch_size, vocab_size]

    return:
        next_token: [batch_size, 1]
    """
    if temperature < 0:
        raise ValueError("temperature must be >= 0.")

    # temperature = 0 时，使用 greedy decoding
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    logits = top_p_filter_logits(logits, top_p=top_p)

    probs = F.softmax(logits, dim=-1)

    next_token = torch.multinomial(probs, num_samples=1)

    return next_token


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token: str,
    context_length: int,
    device: torch.device,
) -> str:
    model.eval()

    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    eos_ids = tokenizer.encode(eos_token)
    if len(eos_ids) != 1:
        raise ValueError(
            f"eos_token={eos_token!r} should encode to exactly one token, got {eos_ids}"
        )
    eos_id = eos_ids[0]

    for _ in range(max_new_tokens):
        # 如果 prompt + 已生成内容超过 context_length，
        # 只保留最后 context_length 个 token 作为模型输入
        model_input = input_ids[:, -context_length:]

        logits = model(model_input)

        # 兼容 model 返回 tuple 的情况，例如 (logits, loss)
        if isinstance(logits, tuple):
            logits = logits[0]

        # logits shape: [batch, seq_len, vocab_size]
        next_logits = logits[:, -1, :]

        next_token = sample_next_token(
            logits=next_logits,
            temperature=temperature,
            top_p=top_p,
        )

        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if next_token.item() == eos_id:
            break

    return tokenizer.decode(input_ids[0].tolist())


def build_model_from_config(config: dict[str, Any]) -> TransformerLM:
    """
    这里的参数名必须和你的 TransformerLM __init__ 对得上。

    如果你的模型参数名不同，就改这里。
    """
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        theta=config.get("rope_theta", 10000.0),
    )

    return model


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)

    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--merges", type=str, required=True)

    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)

    parser.add_argument("--eos-token", type=str, default="<|endoftext|>")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    config = load_json(args.config)

    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges,
        special_tokens=[args.eos_token],
    )

    model = build_model_from_config(config)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = get_state_dict(checkpoint)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    context_length = config["context_length"]

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token=args.eos_token,
        context_length=context_length,
        device=device,
    )

    print(output)


if __name__ == "__main__":
    main()
