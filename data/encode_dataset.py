import pathlib
import numpy as np 
from tokenizers import ByteLevelBPETokenizer 

BASE_DIR = pathlib.Path(__file__).parent.parent
CORPUS_FILE = BASE_DIR / "data" / "code_corpus.txt"
TOKENIZER_DIR = BASE_DIR / "tokenizer"
TRAIN_SPLIT = 0.9

def main():
    tokenizer = ByteLevelBPETokenizer(
        str(TOKENIZER_DIR / "vocab.json"),
        str(TOKENIZER_DIR / "merges.txt"),
    )

    text = CORPUS_FILE.read_text(encoding="utf-8")
    ids = tokenizer.encode(text).ids
    print(f"Total tokens: {len(ids):,}")

    data = np.array(ids, dtype=np.uint16)
    n = int(TRAIN_SPLIT * len(data))
    train_data = data[:n]
    val_data = data[n:]

    train_data.tofile(BASE_DIR / "data" / "train.bin")
    val_data.tofile(BASE_DIR / "data" / "val.bin")

    print(f"train.bin: {len(train_data):,} tokens")
    print(f"val.bin: {len(val_data):,} tokens")

if __name__ == "__main__":
    main()