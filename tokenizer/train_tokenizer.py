import pathlib
from tokenizers import ByteLevelBPETokenizer

CORPUS_FILE = pathlib.Path(__file__).parent.parent / "data" / "code_corpus.txt"
OUTPUT_DIR = pathlib.Path(__file__).parent
VOCAB_SIZE = 8000

def train():
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(CORPUS_FILE)], 
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
    )
    tokenizer.save_model(str(OUTPUT_DIR))
    print(f"Tokenizer saved to {OUTPUT_DIR} (vocab_size={VOCAB_SIZE})")


if __name__ == "__main__":
    train()
