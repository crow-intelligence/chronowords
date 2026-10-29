r"""Turn Wikipedia parquet shards into a plain-text training corpus.

Output format is what ``SVDAlgebra.train`` consumes: one line per paragraph,
tokens separated by single spaces. Lines are context boundaries — skip-grams
never cross them.

Tokenisation (a choice that changes the numbers, so it is stated here and in
the experiment README): Unicode-lowercase, then keep maximal runs of letters
(``[^\W\d_]+``). Digits, punctuation and underscores are dropped. No
lemmatisation. Paragraphs with fewer than ``--min-tokens`` tokens (default 5)
are dropped — mostly headings and list stubs.

Shards are read in path order and articles in file order, so the first N bytes
of the output are a deterministic prefix. ``--max-mb`` stops writing once the
output reaches that size.

Usage::

    uv run --group bench python experiments/scaling/scripts/prepare_wikipedia.py
        data/raw/wikipedia/20231101.hu data/derived/huwiki.txt

(one command, wrapped here)
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq


WORD = re.compile(r"[^\W\d_]+")


def paragraphs(shard_dir: Path):
    """Yield article paragraphs from every parquet shard in ``shard_dir``."""
    shards = sorted(shard_dir.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"No parquet shards in {shard_dir}")
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(columns=["text"], batch_size=2048):
            for text in batch.column(0).to_pylist():
                yield from text.split("\n")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("shard_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--max-mb", type=float, help="stop after this many MB")
    parser.add_argument("--min-tokens", type=int, default=5, help="default: 5")
    args = parser.parse_args()

    limit = int(args.max_mb * 1e6) if args.max_mb else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = lines = tokens = dropped = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for para in paragraphs(args.shard_dir):
            words = WORD.findall(para.lower())
            if len(words) < args.min_tokens:
                dropped += 1
                continue
            line = " ".join(words) + "\n"
            fh.write(line)
            written += len(line.encode("utf-8"))
            lines += 1
            tokens += len(words)
            if lines % 200_000 == 0:
                print(f"\r{written / 1e6:,.0f} MB", end="", file=sys.stderr)
            if limit and written >= limit:
                break
    print(file=sys.stderr)

    record = {
        "source": str(args.shard_dir.relative_to(Path.cwd()))
        if args.shard_dir.is_relative_to(Path.cwd())
        else str(args.shard_dir),
        "output": str(args.out),
        "bytes": written,
        "lines": lines,
        "tokens": tokens,
        "paragraphs_dropped": dropped,
        "params": {
            "tokenizer": WORD.pattern,
            "lowercase": True,
            "min_tokens": args.min_tokens,
            "max_mb": args.max_mb,
        },
    }
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
