#!/usr/bin/env python3
"""Strip PyPI dependency pins from mol_ga wheel so conda-forge supplies RDKit et al."""

from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / "vendor"
SRC = VENDOR / "mol_ga-0.2.1-py3-none-any.whl"
OUT = VENDOR / "mol_ga-0.2.1+conda-py3-none-any.whl"

# Drop runtime Requires-Dist; conda-forge provides numpy, rdkit, joblib.
DROP = {"numpy", "rdkit", "joblib"}


def patch_metadata(text: str) -> str:
    lines = []
    for line in text.splitlines():
        m = re.match(r"Requires-Dist:\s*([^\s;]+)", line)
        if m and m.group(1).lower() in DROP:
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def main() -> None:
    if not SRC.is_file():
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "download", "mol-ga==0.2.1", "--no-deps", "-d", str(VENDOR)]
        )

    work = VENDOR / "_patch_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()

    with zipfile.ZipFile(SRC, "r") as zf:
        zf.extractall(work)

    meta_dir = next(work.glob("mol_ga-*.dist-info"))
    meta_file = meta_dir / "METADATA"
    meta_file.write_text(patch_metadata(meta_file.read_text()))

    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(work.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(work).as_posix())

    shutil.rmtree(work)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
