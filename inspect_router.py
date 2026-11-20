import argparse
import torch
from transformers import AutoTokenizer
from model import MiniDeepSeekMoE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/content/drive/MyDrive/mini_deepseek_checkpoints/mini_deepseek_moe_t4.pt")
    parser.add_argument("--text", default="I am learning machine learning and Python.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(ckpt.get("tokenizer_name", "gpt2"))
    model = MiniDeepSeekMoE(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ids = tokenizer.encode(args.text, return_tensors="pt").to(device)
    with torch.no_grad():
        model(ids)

    tokens = tokenizer.convert_ids_to_tokens(ids[0].tolist())
    for layer_idx, block in enumerate(model.blocks):
        indices = block.moe.last_indices.cpu()
        weights = block.moe.last_weights.cpu()
        print(f"\nTransformer layer {layer_idx + 1}")
        for i, token in enumerate(tokens):
            choices = [
                f"Expert {e}: {w:.3f}"
                for e, w in zip(indices[i].tolist(), weights[i].tolist())
            ]
            print(f"{token:>16} -> " + ", ".join(choices))


if __name__ == "__main__":
    main()
