import argparse
import ast 
import pathlib

import torch
from tokenizers import ByteLevelBPETokenizer

from model import GPT

BASE_DIR = pathlib.Path(__file__).parent 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
BLOCK_SIZE = 256
DROPOUT = 0.1

def load_tokenizer():
    return ByteLevelBPETokenizer(
        str(BASE_DIR / "tokenizer" / "vocab.json"),
        str(BASE_DIR / "tokenizer" / "merges.txt"),
    )

def load_model(checkpoint_path, vocab_size):
    model = GPT(
        vocab_size=vocab_size,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT,
    ).to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def is_valid_python(code_str: str) -> bool:
    try:
        ast.parse(code_str)
        return True
    except SyntaxError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoint_best.pt")
    parser.add_argument("--prompt", default="def ")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    tokenizer = load_tokenizer
    vocab_size = tokenizer.get_vocab_size()
    model = load_model(BASE_DIR / args.checkpoint, vocab_size)

    prompt_ids = tokenizer.encode(args.prompt).ids
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

    valid_count = 0
    for i in range(args.num_samples):
        out_ids = model.generate(
            idx, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k,
        )

        text = tokenizer.decode(out_ids[0].tolist())
        valid = is_valid_python(text)
        valid_count += valid
        print(f"\n--- Sample {i+1} (valid_python={valid}) ---")
        print(text)

    validity_pct = 100 * valid_count / args.num_samples
    print(f"\n\nSyntax validity: {valid_count}/{args.num_samples} ({validity_pct:.1f}%)")

if __name__ == "__main__":
    main()