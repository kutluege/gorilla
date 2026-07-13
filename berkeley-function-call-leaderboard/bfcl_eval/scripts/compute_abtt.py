"""
Fit the ABTT (All-But-The-Top) whitening transform for the Stage 0 governance filter.

One-time offline script. Corpus: BFCL's OWN memory prerequisite-conversation pool
(all five scenarios), segmented at sentence level -- deliberately NOT an external
corpus like Wikipedia/C4, and NOT a single user's memory.

MiniLM (all-MiniLM-L6-v2) embeddings carry a common-mean/anisotropy component that
compresses the discriminative range of cosine similarity (measured on this corpus:
median pairwise cosine 0.096 raw -> ~0.00 after ABTT; the top-16 directions carry
~34% of the variance). ABTT (Mu & Viswanath, 2018) removes the mean and the top-D
covariance eigen-directions;
the result is stored as a fixed transform (mu, U_top) applied at runtime by
bfcl_eval/model_handler/middleware/governance_filter.py.

Usage:
    python bfcl_eval/scripts/compute_abtt.py [--d 16] [--data-dir ...] [--output ...]

The output .npz should be committed so runs are reproducible.
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "bfcl_eval" / "data" / "memory_prereq_conversation"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "bfcl_eval"
    / "model_handler"
    / "middleware"
    / "artifacts"
    / "abtt_minilm_l6_d16.npz"
)
ENCODER_NAME = "all-MiniLM-L6-v2"
MIN_SENTENCE_CHARS = 15

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def collect_sentences(data_dir: Path) -> tuple[list, dict]:
    """All user-message sentences from every prereq conversation, plus per-file
    SHA256 hashes for the reproducibility metadata."""
    sentences: list = []
    file_hashes: dict = {}
    files = sorted(data_dir.glob("*.json"))
    if not files:
        sys.exit(f"No prereq conversation files found in {data_dir}")
    for path in files:
        raw = path.read_bytes()
        file_hashes[path.name] = hashlib.sha256(raw).hexdigest()
        for line in raw.decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            for turn in entry["question"]:
                for message in turn:
                    if message.get("role") != "user":
                        continue
                    for sent in _SENTENCE_SPLIT.split(message["content"]):
                        sent = sent.strip()
                        if len(sent) >= MIN_SENTENCE_CHARS:
                            sentences.append(sent)
    return sentences, file_hashes


def median_pairwise_cosine(X: np.ndarray, sample: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    V = X[idx]
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    sims = V @ V.T
    upper = sims[np.triu_indices_from(sims, k=1)]
    return float(np.median(upper))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=16, help="number of top eigen-directions to remove")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print(f"Collecting sentences from {args.data_dir} ...")
    sentences, file_hashes = collect_sentences(args.data_dir)
    print(f"  {len(sentences)} sentences from {len(file_hashes)} files")

    print(f"Embedding with {ENCODER_NAME} (raw vectors, no normalization) ...")
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(ENCODER_NAME, device="cpu")
    X = np.asarray(
        encoder.encode(sentences, normalize_embeddings=False, show_progress_bar=True),
        dtype=np.float64,
    )
    print(f"  X shape: {X.shape}")

    mu = X.mean(axis=0)
    Xc = X - mu
    Sigma = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    order = np.argsort(eigvals)[::-1]
    u_top = eigvecs[:, order][:, : args.d]  # (dim, d)

    # Sanity check: anisotropy before vs after.
    before = median_pairwise_cosine(X)
    W = Xc - (Xc @ u_top) @ u_top.T
    after = median_pairwise_cosine(W)
    top_var = float(eigvals[order][: args.d].sum() / eigvals.sum())
    print(f"  median pairwise cosine: raw={before:.4f} -> ABTT={after:.4f}")
    print(f"  variance captured by top {args.d} directions: {top_var:.1%}")

    meta = {
        "encoder": ENCODER_NAME,
        "d": args.d,
        "n_sentences": len(sentences),
        "min_sentence_chars": MIN_SENTENCE_CHARS,
        "corpus_sha256": file_hashes,
        "median_cosine_raw": round(before, 4),
        "median_cosine_abtt": round(after, 4),
        "top_d_variance_fraction": round(top_var, 4),
        "created": date.today().isoformat(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        mu=mu,
        u_top=u_top,
        d=np.int64(args.d),
        meta=json.dumps(meta),
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
