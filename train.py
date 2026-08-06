import pathlib
import numpy as np
import torch
from model import GPT

BATCH_SIZE = 32  # how many independent sequences we process in parallel per training step 
BLOCK_SIZE = 256 # how many previous tokens the model can look aat when predicting the next one 
N_EMBD = 384     # width of each token's representation 
N_HEAD = 6       # attention heads (must divide evenly into n_embd)
N_LAYER = 6      # how many blocks stacked 
DROPOUT = 0.1
LEARNING_RATE = 3e-4
MAX_ITERS = 2000      # total training steps 
EVAL_INTERVAL = 250   # how often we pause to measure loss 
EVAL_ITERS = 100      # averaging over multiple batches gives us less noisy loss estimate than one 

BASE_DIR = pathlib.Path(__file__).parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)

def load_data():
    train_data = np.fromfile(BASE_DIR / "data" / "train.bin", dtype=np.uint16)
    val_data = np.fromfile(BASE_DIR / "data" / "val.bin", dtype=np.uint16)
    return torch.from_numpy(train_data.astype(np.int64)), torch.from_numpy(val_data.astype(np.int64))

def get_batch(split, train_data, val_data):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE +1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split, train_data, val_data)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out 

def get_vocab_size():
    import json
    with open(BASE_DIR / "tokenizer" / "vocab.json") as f:
        vocab = json.load(f)
    return len(vocab)

def main():
    print(f"Using device: {DEVICE}")
    train_data, val_data = load_data()
    print(f"train tokens: {len(train_data):,} | val tokens: {len(val_data):,}")

    vocab_size = get_vocab_size()
    print(f"vocab_size: {vocab_size}")

    model = GPT(
        vocab_size=vocab_size,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for it in range(MAX_ITERS):
        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch("train", train_data, val_data)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), BASE_DIR / "checkpoint_final.pt")
    print("Training complete. Saved checkpoint_final.pt")

if __name__ == "__main__":
    main()