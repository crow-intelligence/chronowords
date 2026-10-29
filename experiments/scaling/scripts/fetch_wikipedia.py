"""Download Wikipedia plain-text parquet shards from Hugging Face.

Source: the ``wikimedia/wikipedia`` dataset (CC BY-SA 4.0 / GFDL), which ships
article text already stripped of wiki markup, so no markup parser is needed.

Shards land untouched in ``data/raw/wikipedia/<snapshot>.<lang>/`` (gitignored).
Next to each shard a ``.manifest.json`` records URL, size, SHA-256 and fetch
time; the manifests are committed as the provenance record.

Usage::

    uv run python experiments/scaling/scripts/fetch_wikipedia.py hu
    uv run python experiments/scaling/scripts/fetch_wikipedia.py en --shards 0-7
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import UTC
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
RAW_DIR = REPO / "data" / "raw" / "wikipedia"
API = "https://huggingface.co/api/datasets/wikimedia/wikipedia/tree/main/{config}"
RESOLVE = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/{path}"
CHUNK = 1 << 20


def list_shards(config: str) -> list[dict]:
    """Return the parquet file entries for ``config``, sorted by path."""
    with urllib.request.urlopen(API.format(config=config)) as resp:
        entries = json.load(resp)
    shards = [e for e in entries if e["path"].endswith(".parquet")]
    if not shards:
        raise SystemExit(f"No parquet shards found for {config}")
    return sorted(shards, key=lambda e: e["path"])


def parse_range(spec: str | None, n: int) -> list[int]:
    """Parse ``"0-7"`` / ``"3"`` into shard indices; ``None`` means all."""
    if spec is None:
        return list(range(n))
    lo, _, hi = spec.partition("-")
    indices = list(range(int(lo), int(hi or lo) + 1))
    if indices[-1] >= n:
        raise SystemExit(f"Shard range {spec} out of bounds (have {n})")
    return indices


def download(entry: dict, out_dir: Path) -> None:
    """Download one shard (skipping if already complete) and write its manifest."""
    target = out_dir / Path(entry["path"]).name
    manifest = target.with_suffix(".manifest.json")
    if manifest.exists() and target.exists():
        if target.stat().st_size == entry["size"]:
            print(f"skip {target.name} (already fetched)")
            return

    url = RESOLVE.format(path=entry["path"])
    sha = hashlib.sha256()
    tmp = target.with_suffix(".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as fh:
        done = 0
        while chunk := resp.read(CHUNK):
            fh.write(chunk)
            sha.update(chunk)
            done += len(chunk)
            print(f"\r{target.name}: {done / 1e6:,.0f} MB", end="", file=sys.stderr)
    print(file=sys.stderr)

    if tmp.stat().st_size != entry["size"]:
        raise SystemExit(f"Size mismatch for {target.name}; left {tmp} in place")
    expected = entry.get("lfs", {}).get("oid")
    if expected and expected != sha.hexdigest():
        raise SystemExit(f"SHA-256 mismatch for {target.name}; left {tmp} in place")
    tmp.rename(target)

    manifest.write_text(
        json.dumps(
            {
                "source": "huggingface.co/datasets/wikimedia/wikipedia",
                "licence": "CC BY-SA 4.0 / GFDL",
                "url": url,
                "size": entry["size"],
                "sha256": sha.hexdigest(),
                "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"done {target.name}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("lang", help="Wikipedia language code, e.g. 'hu' or 'en'")
    parser.add_argument("--snapshot", default="20231101", help="default: 20231101")
    parser.add_argument("--shards", help="index range like '0-7'; default: all")
    args = parser.parse_args()

    config = f"{args.snapshot}.{args.lang}"
    shards = list_shards(config)
    out_dir = RAW_DIR / config
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in parse_range(args.shards, len(shards)):
        download(shards[i], out_dir)


if __name__ == "__main__":
    main()
