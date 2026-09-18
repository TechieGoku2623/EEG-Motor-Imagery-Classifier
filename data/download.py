"""Download EEGMMIDB EDF files from PhysioNet using wfdb."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import wfdb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def subject_edf_files(subjects: list[int], runs: list[int]) -> list[str]:
    """Relative paths of EDF files in the PhysioNet eegmmidb tree."""
    files = []
    for subject in subjects:
        for run in runs:
            files.append(f"S{subject:03d}/S{subject:03d}R{run:02d}.edf")
    return files


MIN_EDF_BYTES = 100_000  # incomplete downloads are typically tiny


def _is_complete(path: Path) -> bool:
    return path.exists() and path.stat().st_size >= MIN_EDF_BYTES


def _download_one(rel: str, out_dir: Path) -> str:
    dest = out_dir / rel
    if _is_complete(dest):
        return rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    # EEGMMIDB is EDF+, not WFDB header/dat files.
    wfdb.dl_files(
        db=config.PHYSIONET_DB,
        dl_dir=str(out_dir),
        files=[rel],
        keep_subdirs=True,
        overwrite=True,
    )
    if not _is_complete(dest):
        raise RuntimeError(f"Incomplete download: {rel}")
    return rel


def download(subjects: list[int], runs: list[int], out_dir: Path, workers: int = 8) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = subject_edf_files(subjects, runs)
    missing = [f for f in files if not _is_complete(out_dir / f)]
    if not missing:
        print(f"All {len(files)} EDF files already present in {out_dir}")
        return

    print(
        f"Downloading {len(missing)} / {len(files)} EDF files "
        f"from PhysioNet {config.PHYSIONET_DB} via wfdb.dl_files ({workers} workers)"
    )
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_download_one, rel, out_dir): rel for rel in missing}
        for fut in as_completed(futs):
            rel = futs[fut]
            try:
                fut.result()
                ok += 1
                if ok % 10 == 0 or ok == len(missing):
                    print(f"  {ok}/{len(missing)} done (last: {rel})", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED {rel}: {exc}", flush=True)
    print("Download complete.", flush=True)


def parse_subjects(spec: str) -> list[int]:
    subjects: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            subjects.extend(range(int(a), int(b) + 1))
        else:
            subjects.append(int(part))
    return sorted(set(subjects))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download EEGMMIDB EDF files via wfdb")
    parser.add_argument(
        "--subjects",
        default="1-20",
        help="Comma-separated ids or ranges, e.g. 1-20 or 1,2,5",
    )
    parser.add_argument(
        "--runs",
        default="3-14",
        help="Run numbers to download (default motor runs 3-14)",
    )
    parser.add_argument("--out", type=Path, default=config.DATA_RAW)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    subjects = parse_subjects(args.subjects)
    runs = parse_subjects(args.runs)
    download(subjects, runs, args.out, workers=args.workers)


if __name__ == "__main__":
    main()
