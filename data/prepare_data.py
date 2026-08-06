import pathlib
import subprocess

REPOS = [
    "https://github.com/psf/requests.git",
    "https://github.com/pallets/flask.git",
    "https://github.com/encode/httpx.git",
    "https://github.com/Textualize/rich.git",
    "https://github.com/pydantic/pydantic.git",
    "https://github.com/tiangolo/fastapi.git",
]
RAW_DIR = pathlib.Path(__file__).parent / "raw_repos"

def clone_repos():
    RAW_DIR.mkdir(exist_ok=True)
    for url in REPOS:
        name = url.rstrip("/").split("/")[-1].replace(".git", "")
        dest = RAW_DIR / name 
        if dest.exists():
            print(f"[skip] {name} already cloned")
            continue
        print(f"[clone] {name}")
        subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)




SKIP_PATTERNS = ["/tests/", "/test/", "/vendor", "/migrations/", "/.git", "/build/", "/dist/"]
OUTPUT_FILE = pathlib.Path(__file__).parent / "code_corpus.txt"

def should_skip(path: pathlib.Path) -> bool:
    p = str(path)
    return any(pattern in p for pattern in SKIP_PATTERNS)

def build_corpus():
    files = [f for f in RAW_DIR.rglob("*.py") if not should_skip(f)]
    print(f"Found {len(files)} python files (after filtering)")

    written = 0
    with open (OUTPUT_FILE, "w", encoding="utf-8") as out:
        for f in files:
            try:
                text = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue
            if len(text.strip()) == 0:
                continue
            out.write(text)
            out.write("\n\n")
            written += 1

    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"Wrote {written} files -> {OUTPUT_FILE} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    clone_repos()
    build_corpus()