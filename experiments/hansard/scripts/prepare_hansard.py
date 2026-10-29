"""Extract House of Commons speech text from TheyWorkForYou XML, one file per decade.

Input: the untouched ``data/raw/hansard/debates/debates*.xml`` mirror (see
``fetch_hansard.sh``). A sitting day can have several versions
(``debatesYYYY-MM-DDa.xml``, ``...b.xml``); only the file whose root says
``latest="yes"`` is used. If a day has no such file, its last version letter is
used.

Output: ``data/derived/hansard/commons_<decade>s.txt`` (gitignored), one line
per ``<p>`` paragraph inside a ``<speech>``, with inner markup removed and
entities decoded. Headings, tables and division lists are not included. Text
is left raw (case and punctuation kept) so that tokenisation is the
library's choice, made at training time.

A file that fails to parse is skipped and counted, never silently dropped:
the per-decade record (``.json`` next to each text file) lists them.

Usage::

    uv run python experiments/hansard/scripts/prepare_hansard.py
"""

import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "data" / "raw" / "hansard" / "debates"
OUT = REPO / "data" / "derived" / "hansard"
NAME = re.compile(r"debates(\d{4})-(\d\d)-(\d\d)([a-z])\.xml$")
SPACE = re.compile(r"\s+")


def is_latest(path: Path) -> bool | None:
    """Read the ``latest`` attribute from the root element, if present."""
    with path.open("rb") as fh:
        head = fh.read(400).decode("latin-1")
    m = re.search(r'latest="(yes|no)"', head)
    return None if m is None else m.group(1) == "yes"


def pick_versions(files: list[Path]) -> list[Path]:
    """One file per sitting day: the ``latest="yes"`` one, else the last letter."""
    by_day: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        m = NAME.search(f.name)
        if m:
            by_day["-".join(m.groups()[:3])].append(f)
    chosen = []
    for day in sorted(by_day):
        versions = sorted(by_day[day])
        latest = [f for f in versions if is_latest(f)]
        chosen.append(latest[-1] if latest else versions[-1])
    return chosen


def paragraphs(path: Path) -> tuple[str, list[str], str | None]:
    """Speech paragraphs of one file, or an error message if it fails to parse."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        return path.name, [], str(exc)
    out = []
    for speech in root.iter("speech"):
        for p in speech.iter("p"):
            text = SPACE.sub(" ", "".join(p.itertext())).strip()
            if text:
                out.append(text)
    return path.name, out, None


def main() -> None:
    """CLI entry point."""
    files = sorted(RAW.glob("debates*.xml"))
    if not files:
        sys.exit(f"No debates*.xml under {RAW}; run fetch_hansard.sh first")
    chosen = pick_versions(files)
    OUT.mkdir(parents=True, exist_ok=True)

    decades: dict[str, list[Path]] = defaultdict(list)
    for f in chosen:
        decades[f.name[7:10] + "0s"].append(f)

    for decade, day_files in sorted(decades.items()):
        target = OUT / f"commons_{decade}.txt"
        record = {
            "decade": decade,
            "sitting_days": len(day_files),
            "first": day_files[0].name,
            "last": day_files[-1].name,
            "paragraphs": 0,
            "bytes": 0,
            "parse_errors": {},
        }
        with (
            ProcessPoolExecutor() as pool,
            target.open("w", encoding="utf-8") as fh,
        ):
            for name, paras, error in pool.map(paragraphs, day_files, chunksize=16):
                if error:
                    record["parse_errors"][name] = error
                    continue
                for para in paras:
                    line = para + "\n"
                    fh.write(line)
                    record["paragraphs"] += 1
                    record["bytes"] += len(line.encode("utf-8"))
        record["source"] = str(RAW.relative_to(REPO))
        record["output"] = str(target.relative_to(REPO))
        target.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
        print(
            f"{decade}: {record['sitting_days']:5d} days  {record['paragraphs']:9,d} "
            f"paragraphs  {record['bytes'] / 1e6:8.1f} MB  "
            f"parse errors: {len(record['parse_errors'])}"
        )


if __name__ == "__main__":
    main()
