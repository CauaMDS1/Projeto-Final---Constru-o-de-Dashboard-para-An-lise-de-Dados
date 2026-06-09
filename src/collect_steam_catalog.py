from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from src.config import STEAM_CATALOG_API_CSV, ensure_directories


STEAMSPY_ALL_URL = "https://steamspy.com/api.php?request=all&page={page}"


def collect_steam_catalog(
    output_path: Path = STEAM_CATALOG_API_CSV,
    pages: int = 1,
    pause_seconds: float = 0.5,
) -> pd.DataFrame:
    """Collect a public SteamSpy catalog snapshot for the bonus requirement."""
    ensure_directories()
    collected: list[pd.DataFrame] = []
    collected_at = datetime.now(timezone.utc).isoformat()

    for page in range(pages):
        url = STEAMSPY_ALL_URL.format(page=page)
        with urlopen(url, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))

        if not payload:
            break

        page_frame = pd.DataFrame(payload.values())
        page_frame["data_coleta_utc"] = collected_at
        page_frame["source_api"] = "steamspy"
        page_frame["source_page"] = page
        collected.append(page_frame)

        if page < pages - 1:
            time.sleep(pause_seconds)

    if not collected:
        raise RuntimeError("No rows returned by the SteamSpy API.")

    frame = pd.concat(collected, ignore_index=True)
    frame = frame.drop_duplicates(subset=["appid"])
    frame.to_csv(output_path, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Steam app catalog via API.")
    parser.add_argument(
        "--output",
        type=Path,
        default=STEAM_CATALOG_API_CSV,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=1,
        help="Number of SteamSpy pages to collect. Each page returns a catalog slice.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.5,
        help="Pause between API pages.",
    )
    args = parser.parse_args()
    frame = collect_steam_catalog(args.output, args.pages, args.pause_seconds)
    print(f"Collected {len(frame):,} apps from SteamSpy API into {args.output}")


if __name__ == "__main__":
    main()
