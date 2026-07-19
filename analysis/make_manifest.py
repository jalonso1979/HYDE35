"""Regenerate MANIFEST.md (OSF deposit manifest).

Reproduces the structure of the 2026-05-14 hand-generated manifest:
SHA-256 + bytes for top-level files, paper sources, and pipeline scripts;
bytes only for generated data and figures. Run from the repo root:

    python -m analysis.make_manifest
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

ROOT = Path("/Volumes/BIGDATA/HYDE35")
OUT = ROOT / "MANIFEST.md"

TOP_FILES = ["README.md", "REPRODUCE.md", "CHANGELOG.md", "LICENSE",
             "CITATION.cff", "Makefile", "requirements.txt", "pyproject.toml"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hashed_section(title: str, files: list[Path]) -> list[str]:
    lines = [f"## {title}", "", "| File | Bytes | SHA-256 |", "|---|---:|---|"]
    for p in files:
        rel = p.relative_to(ROOT)
        lines.append(f"| {rel} | {p.stat().st_size} | `{sha256(p)}` |")
    lines.append("")
    return lines


def sized_section(title: str, files: list[Path]) -> tuple[list[str], int]:
    lines = [f"## {title}", "", "| File | Bytes |", "|---|---:|"]
    total = 0
    for p in files:
        total += p.stat().st_size
        lines.append(f"| {p.relative_to(ROOT)} | {p.stat().st_size} |")
    lines.append("")
    return lines, total


def main() -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = ["# OSF deposit manifest", "",
             f"Generated {ts} by `analysis/make_manifest.py`.",
             "All file sizes in bytes. Hashes are SHA-256 of file content.", ""]

    lines += hashed_section(
        "Top-level files", [ROOT / f for f in TOP_FILES if (ROOT / f).exists()])
    lines += hashed_section(
        "Paper", sorted((ROOT / "paper").glob("*.tex"))
        + sorted((ROOT / "paper").glob("*.pdf"))
        + sorted((ROOT / "paper").glob("*.bib")))
    lines += hashed_section(
        "Build scripts (analysis/shared/)",
        sorted((ROOT / "analysis" / "shared").glob("*.py")))
    lines += hashed_section(
        "Analysis scripts (analysis/paper4_shadow/)",
        sorted((ROOT / "analysis" / "paper4_shadow").glob("*.py")))

    total = 0
    for p in TOP_FILES:
        f = ROOT / p
        if f.exists():
            total += f.stat().st_size

    data_files = sorted((ROOT / "analysis" / "data").glob("*.parquet")) + \
        sorted((ROOT / "analysis" / "data").glob("*.csv")) + \
        sorted((ROOT / "analysis" / "data" / "long_shadow_fertility").glob("*.parquet")) + \
        sorted((ROOT / "analysis" / "data" / "era5_derived").rglob("*.parquet"))
    sec, sz = sized_section("Generated data (analysis/data/)", data_files)
    lines += sec
    total += sz

    fig_files = []
    for d in ("paper4_v2", "paper4", "long_shadow_fertility"):
        fd = ROOT / "analysis" / "figures" / d
        if fd.exists():
            fig_files += sorted(fd.glob("*.pdf")) + sorted(fd.glob("*.png"))
    sec, sz = sized_section("Generated figures (analysis/figures/)", fig_files)
    lines += sec
    total += sz

    lines += ["## Total package size", "",
              f"Approximate redistributable total: {total} bytes", ""]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT} ({len(lines)} lines, total {total:,} bytes listed)")


if __name__ == "__main__":
    main()
