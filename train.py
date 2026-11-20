import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from model import MiniDeepSeekMoE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_stories", type=int, default=100_000)
    p.add_argument("--max_steps", type=int, default=2_000)
    p.add_argument("--batch_size", type=int, default=4,
                   help="Micro-batch size per GPU step.")
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--save_dir", type=str, default="/content/drive/MyDrive/mini_deepseek_checkpoints")
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    return p.parse_args()


def build_token_stream(num_stories, tokenizer):
    print(f"Loading TinyStories train[:{num_stories}] ...")
    ds = load_dataset(
        "roneneldan/TinyStories",
        split=f"train[:{num_stories}]"
    )

    eos_id = tokenizer.eos_token_id

    def tokenize_batch(batch):
        encoded = tokenizer(
            batch["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=512
        )["input_ids"]
        return {"ids": [row + [eos_id] for row in encoded]}

    tokenized = ds.map(
        tokenize_batch,
        batched=True,
        remove_columns=ds.column_names,
        desc="Tokenizing TinyStories"
    )

    arrays = [
        np.asarray(row, dtype=np.uint32)
        for row in tokenized["ids"] if len(row) > 1
    ]
    flat = np.concatenate(arrays)
    print(f"Token count: {len(flat):,}")
    # int32 halves host RAM vs int64; IDs are converted to int64 on GPU.
    return torch.from_numpy(flat.astype(np.int32, copy=False))


def get_batch(data, batch_size, seq_len, device):
    max_start = len(data) - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,))
    offsets = torch.arange(seq_len)
    x = data[starts[:, None] + offsets[None, :]].to(device=device, dtype=torch.long)
    y = data[starts[:, None] + offsets[None, :] + 1].to(device=device, dtype=torch.long)
    return x, y


@torch.no_grad()
def evaluate(model, train_data, val_data, args, device, amp_enabled, amp_dtype,
             aux_coef, batches=20):
    model.eval()
    metrics = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(batches):
            x, y = get_batch(data, args.batch_size, args.seq_len, device)
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
            ):
                logits, aux = model(x)
            lm = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )
            losses.append((lm + aux_coef * aux.float()).item())
        metrics[name] = sum(losses) / len(losses)
    model.train()
    return metrics


def save_checkpoint(path, model, optimizer, scaler, config, step, val_loss):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config,
        "step": step,
        "val_loss": val_loss,
        "tokenizer_name": "gpt2"
    }, path)


def main():
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not found. In Colab choose Runtime > Change runtime type > T4 GPU."
        )

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | VRAM: {props.total_memory / 1024**3:.1f} GB")
    if "T4" not in props.name:
        print("Note: this configuration was tuned for a T4; adjust batch size if needed.")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    token_stream = build_token_stream(args.num_stories, tokenizer)
    split = int(0.95 * len(token_stream))
    train_data, val_data = token_stream[:split], token_stream[split:]

    config = {
        "vocab_size": len(tokenizer),
        "d_model": 256,
        "n_layers": 6,
        "n_heads": 8,
        "d_ff": 512,
        "num_experts": 4,
        "top_k": 2,
        "max_seq_len": args.seq_len,
        "sliding_window": 64,
        "dropout": 0.0,
        "gradient_checkpointing": not args.no_gradient_checkpointing
    }

    model = MiniDeepSeekMoE(**config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params:,} ({params / 1e6:.2f}M)")
    print(f"Micro-batch={args.batch_size}, accumulation={args.grad_accum}, "
          f"effective batch={args.batch_size * args.grad_accum}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        betas=(0.9, 0.95), weight_decay=0.1
    )

    amp_enabled = True
    amp_dtype = torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    aux_coef = 0.01

    save_path = Path(args.save_dir) / "mini_deepseek_moe_t4.pt"
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, args.max_steps + 1):
        accumulated_lm = 0.0
        accumulated_aux = 0.0

        for micro_step in range(args.grad_accum):
            x, y = get_batch(data=train_data, batch_size=args.batch_size,
                             seq_len=args.seq_len, device=device)

            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
            ):
                logits, aux_loss = model(x)

            # Compute cross entropy in FP32 for numerical stability.
            lm_loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )
            loss = (lm_loss + aux_coef * aux_loss.float()) / args.grad_accum
            scaler.scale(loss).backward()

            accumulated_lm += lm_loss.detach().item()
            accumulated_aux += aux_loss.detach().item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % 25 == 0:
            print(
                f"step {step:5d}/{args.max_steps} | "
                f"lm_loss {accumulated_lm / args.grad_accum:.4f} | "
                f"aux {accumulated_aux / args.grad_accum:.4f} | "
                f"lr {optimizer.param_groups[0]['lr']:.2e}"
            )

        if step % args.eval_interval == 0:
            metrics = evaluate(
                model, train_data, val_data, args, device,
                amp_enabled, amp_dtype, aux_coef
            )
            val_ppl = math.exp(min(metrics["val"], 20))
            print(
                f"[eval] step={step} train={metrics['train']:.4f} "
                f"val={metrics['val']:.4f} perplexity={val_ppl:.2f}"
            )
            save_checkpoint(
                save_path, model, optimizer, scaler, config,
                step, metrics["val"]
            )
            print("Checkpoint saved:", save_path)

        elif step % args.save_interval == 0:
            save_checkpoint(
                save_path, model, optimizer, scaler, config,
                step, float("nan")
            )
            print("Checkpoint saved:", save_path)

    save_checkpoint(
        save_path, model, optimizer, scaler, config,
        args.max_steps, float("nan")
    )
    print("Training finished. Final checkpoint:", save_path)


if __name__ == "__main__":
    main()
