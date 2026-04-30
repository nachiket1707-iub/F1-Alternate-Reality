"""
fetch_2024_data.py
------------------
Fetches 2024 F1 season race results and qualifying positions.
Used to train the ML championship predictor in preprocess.py.
Only results are fetched — no telemetry — so this runs in ~3-5 minutes.

Usage:
    python data/fetch_2024_data.py            # fetch all 24 rounds
    python data/fetch_2024_data.py --force    # re-fetch even if files exist
"""

import fastf1
import pandas as pd
import sys
import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR    = PROJECT_ROOT / "cache"
RAW_2024_DIR = PROJECT_ROOT / "data" / "raw" / "2024"
SEASON_YEAR  = 2024
TOTAL_ROUNDS = 24


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def is_already_fetched(round_num: int) -> bool:
    return (
        (RAW_2024_DIR / f"race_{round_num:02d}.csv").exists() and
        (RAW_2024_DIR / f"quali_{round_num:02d}.csv").exists()
    )


def fetch_round(round_num: int, force: bool, log: logging.Logger) -> bool:
    if not force and is_already_fetched(round_num):
        log.info(f"Round {round_num:02d}: already fetched — skipping.")
        return True

    log.info(f"Round {round_num:02d}: fetching...")

    try:
        race_session = fastf1.get_session(SEASON_YEAR, round_num, "R")
        race_session.load(telemetry=False, weather=False, messages=False)

        res = race_session.results.copy()
        race_df = pd.DataFrame({
            "round_number":  round_num,
            "event_name":    race_session.event["EventName"],
            "driver_code":   res["Abbreviation"],
            "team_name":     res["TeamName"],
            "grid_position": pd.array(res["GridPosition"], dtype="Int64"),
            "position":      pd.array(res["Position"],     dtype="Int64"),
            "points":        res["Points"].astype(float),
            "status":        res["Status"],
        })
        race_df.to_csv(RAW_2024_DIR / f"race_{round_num:02d}.csv", index=False)
        log.info(f"  Round {round_num:02d}: race results saved ({len(race_df)} drivers)")
    except Exception as e:
        log.warning(f"  Round {round_num:02d}: race session failed — {type(e).__name__}: {e}")
        return False

    try:
        qual_session = fastf1.get_session(SEASON_YEAR, round_num, "Q")
        qual_session.load(telemetry=False, weather=False, messages=False)

        qres = qual_session.results.copy()
        qual_df = pd.DataFrame({
            "round_number": round_num,
            "event_name":   qual_session.event["EventName"],
            "driver_code":  qres["Abbreviation"],
            "team_name":    qres["TeamName"],
            "position":     pd.array(qres["Position"], dtype="Int64"),
        })
        qual_df.to_csv(RAW_2024_DIR / f"quali_{round_num:02d}.csv", index=False)
        log.info(f"  Round {round_num:02d}: qualifying results saved")
    except Exception as e:
        log.warning(f"  Round {round_num:02d}: qualifying session failed — {e}")

    log.info(f"Round {round_num:02d}: done.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fetch 2024 F1 season results for ML training"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-fetch even if files exist",
    )
    args = parser.parse_args()

    log = setup_logging()
    RAW_2024_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    log.info(f"Fetching 2024 F1 season results ({TOTAL_ROUNDS} rounds)...")
    log.info(f"Output: {RAW_2024_DIR}")

    success, failed = 0, []
    for round_num in range(1, TOTAL_ROUNDS + 1):
        ok = fetch_round(round_num, force=args.force, log=log)
        if ok:
            success += 1
        else:
            failed.append(round_num)

    log.info(f"Fetch complete: {success}/{TOTAL_ROUNDS} rounds processed.")
    if failed:
        log.warning(f"Failed rounds: {failed}")


if __name__ == "__main__":
    main()
