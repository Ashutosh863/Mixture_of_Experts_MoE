import argparse
import torch
from transformers import AutoTokenizer
from model import MiniDeepSeekMoE


def sample_token(logits, temperature=0.8, top_k=40):
    logits = logits / max(temperature, 1e-5)
    if top_k and top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < values[-1], float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/content/drive/MyDrive/mini_deepseek_checkpoints/mini_deepseek_moe_t4.pt")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(ckpt.get("tokenizer_name", "gpt2"))
    model = MiniDeepSeekMoE(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            context = ids[:, -model.max_seq_len:]
            with torch.autocast(
                device_type="cuda", dtype=torch.float16,
                enabled=(device == "cuda")
            ):
                logits, _ = model(context)
            next_id = sample_token(
                logits[0, -1].float(), args.temperature, args.top_k
            ).view(1, 1)
            ids = torch.cat([ids, next_id], dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break

    print(tokenizer.decode(ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
