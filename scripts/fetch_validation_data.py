"""Download the BBBC039 validation dataset.

    python scripts/fetch_validation_data.py

BBBC039 is 200 fluorescence fields of Hoechst-stained U2OS nuclei with roughly
23,000 manually annotated objects, released CC0 by the Broad Bioimage Benchmark
Collection. About 76 MB. The data is not committed to the repository.

Citation:
    image set BBBC039v1, Caicedo et al. 2018, available from the Broad Bioimage
    Benchmark Collection [Ljosa et al., Nature Methods, 2012].
"""

from __future__ import annotations

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "validation_data" / "bbbc039"
BASE_URL = "https://data.broadinstitute.org/bbbc/BBBC039"
ARCHIVES = ("images", "masks", "metadata")


def download(name: str, destination: Path) -> None:
    url = f"{BASE_URL}/{name}.zip"
    archive = destination / f"{name}.zip"
    if archive.exists():
        print(f"  {name}.zip already downloaded")
    else:
        print(f"  downloading {url}")
        with urllib.request.urlopen(url, timeout=600) as response:
            archive.write_bytes(response.read())
        print(f"    {archive.stat().st_size / 1_048_576:.0f} MB")

    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
    print(f"  extracted {name}")


def main() -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    print(f"Fetching BBBC039 into {TARGET}")

    for name in ARCHIVES:
        try:
            download(name, TARGET)
        except Exception as exc:  # noqa: BLE001 - network failures are expected
            print(f"  failed to fetch {name}: {exc}", file=sys.stderr)
            return 1

    # The archives carry macOS resource forks that would otherwise be globbed
    # alongside the real images.
    junk = TARGET / "__MACOSX"
    if junk.exists():
        shutil.rmtree(junk)

    images = len(list((TARGET / "images").glob("*.tif")))
    masks = len(list((TARGET / "masks").glob("*.png")))
    print(f"\nReady: {images} images, {masks} masks")
    if images != 200 or masks != 200:
        print("warning: expected 200 of each", file=sys.stderr)
        return 1

    print("Now run:  python scripts/validate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
