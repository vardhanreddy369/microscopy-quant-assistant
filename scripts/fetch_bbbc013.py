"""Download BBBC013, a real two-channel marker-translocation assay.

BBBC013 images the FKHR-GFP nuclear-translocation assay: channel 1 is the marker
(FKHR-GFP), channel 2 is the nuclei (DNA). Drug treatment drives the marker into
the nucleus, so the fraction of cells with high nuclear marker is a real
percent-positive readout with a known dose gradient — exactly the shape the
positivity method calls, on real data rather than simulation.

    python scripts/fetch_bbbc013.py

About 31 MB, released CC BY 3.0 by the Broad Bioimage Benchmark Collection.
Not committed to the repository.

Citation:
    Ilya Ravkin, image set BBBC013v1, from the Broad Bioimage Benchmark
    Collection [Ljosa et al., Nature Methods, 2012].
"""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "validation_data" / "bbbc013"
BASE = "https://data.broadinstitute.org/bbbc/BBBC013"
PLATEMAPS = ("all", "wortmannin", "ly294002")


def download(url: str, path: Path) -> None:
    if path.exists():
        print(f"  {path.name} already present")
        return
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=600) as response:
        path.write_bytes(response.read())


def main() -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    print(f"Fetching BBBC013 into {TARGET}")

    archive = TARGET / "images.zip"
    try:
        download(f"{BASE}/BBBC013_v1_images_bmp.zip", archive)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(TARGET)
        for name in PLATEMAPS:
            download(f"{BASE}/BBBC013_v1_platemap_{name}.txt",
                     TARGET / f"platemap_{name}.txt")
    except Exception as exc:  # noqa: BLE001 - network failures expected
        print(f"  failed: {exc}", file=sys.stderr)
        return 1

    images = list((TARGET / "BBBC013_v1_images_bmp").glob("*.BMP"))
    print(f"\nReady: {len(images)} images ({len(images) // 2} wells x 2 channels)")
    if len(images) != 192:
        print("warning: expected 192 images", file=sys.stderr)
        return 1
    print("Now run:  python scripts/validate_bbbc013.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
