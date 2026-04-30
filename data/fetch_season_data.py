"""
fetch_season_data.py
--------------------
Fetches 2025 F1 season data using FastF1 and saves to data/raw/.
Run this script first, before preprocess.py.

Usage:
    python data/fetch_season_data.py                  # fetch all rounds
    python data/fetch_season_data.py --rounds 1 5     # fetch rounds 1-5 only
    python data/fetch_season_data.py --force           # re-fetch even if cached
"""

import fastf1
import pandas as pd
import json
import sys
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR    = PROJECT_ROOT / "cache"
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR  = RAW_DIR / "results"
TELEM_DIR    = RAW_DIR / "telemetry"
META_DIR     = RAW_DIR / "session_meta"
SEASON_YEAR  = 2025

# ── 2025 Race Schedule ────────────────────────────────────────────────────────
RACE_SCHEDULE_2025 = [
    (1,  "Australian Grand Prix"),
    (2,  "Chinese Grand Prix"),
    (3,  "Japanese Grand Prix"),
    (4,  "Bahrain Grand Prix"),
    (5,  "Saudi Arabian Grand Prix"),
    (6,  "Miami Grand Prix"),
    (7,  "Emilia Romagna Grand Prix"),
    (8,  "Monaco Grand Prix"),
    (9,  "Spanish Grand Prix"),
    (10, "Canadian Grand Prix"),
    (11, "Austrian Grand Prix"),
    (12, "British Grand Prix"),
    (13, "Belgian Grand Prix"),
    (14, "Hungarian Grand Prix"),
    (15, "Dutch Grand Prix"),
    (16, "Italian Grand Prix"),
    (17, "Azerbaijan Grand Prix"),
    (18, "Singapore Grand Prix"),
    (19, "United States Grand Prix"),
    (20, "Mexico City Grand Prix"),
    (21, "São Paulo Grand Prix"),
    (22, "Las Vegas Grand Prix"),
    (23, "Qatar Grand Prix"),
    (24, "Abu Dhabi Grand Prix"),
]


# ── Setup ─────────────────────────────────────────────────────────────────────
def setup_directories():
    for d in [CACHE_DIR, RESULTS_DIR, TELEM_DIR, META_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def setup_fastf1_cache():
    fastf1.Cache.enable_cache(str(CACHE_DIR))


def setup_logging() -> logging.Logger:
    log_path = PROJECT_ROOT / "fetch.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
    )
    return logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────
def event_name_to_slug(event_name: str) -> str:
    """'Australian Grand Prix' → 'Australian_GP'"""
    return event_name.replace(" Grand Prix", "").replace(" ", "_") + "_GP"


def is_race_already_fetched(round_num: int, slug: str) -> bool:
    race_file = RESULTS_DIR / f"race_{round_num:02d}_{slug}.csv"
    qual_file = RESULTS_DIR / f"qualifying_{round_num:02d}_{slug}.csv"
    tele_file = TELEM_DIR   / f"fastest_lap_{round_num:02d}_{slug}.parquet"
    meta_file = META_DIR    / f"meta_{round_num:02d}_{slug}.json"
    return all(f.exists() for f in [race_file, qual_file, tele_file, meta_file])


# ── Session loading ───────────────────────────────────────────────────────────
def load_session_safe(year: int, round_num: int, session_type: str,
                      logger: logging.Logger):
    """Load a FastF1 session, returning (session, ok). Returns (None, False) on any error."""
    try:
        session = fastf1.get_session(year, round_num, session_type)

        # Determine race date to skip future rounds
        event = session.event
        race_date = event.get("Session5Date") or event.get("EventDate")
        if race_date is not None and not pd.isnull(race_date):
            if hasattr(race_date, "date"):
                race_date = race_date.date()
            if race_date > datetime.now(timezone.utc).date():
                logger.info(f"  Round {round_num}: race date {race_date} is in the future — skipping.")
                return None, False

        load_kwargs = dict(messages=False)
        if session_type == "R":
            load_kwargs.update(telemetry=True, weather=True)
        else:
            load_kwargs.update(telemetry=False, weather=False)

        session.load(**load_kwargs)
        return session, True

    except Exception as e:
        logger.warning(f"  Round {round_num} [{session_type}]: {type(e).__name__}: {e}")
        return None, False


# ── Extraction ────────────────────────────────────────────────────────────────
def extract_race_results(session, round_num: int) -> pd.DataFrame:
    res = session.results.copy()

    # Fastest lap info lives in session.laps in FastF1 3.8.x (not in results)
    fastest_lap_driver = None
    fastest_lap_time   = None
    try:
        laps = session.laps
        if not laps.empty:
            valid = laps.pick_wo_box()
            if valid.empty:
                valid = laps
            fl = valid.pick_fastest()
            fastest_lap_driver = fl["Driver"]
            fastest_lap_time   = str(fl["LapTime"])
    except Exception:
        pass

    # Build per-driver fastest_lap_rank (1 for the holder, NA for everyone else)
    driver_codes = res["Abbreviation"].tolist()
    fl_ranks = [
        1 if d == fastest_lap_driver else pd.NA
        for d in driver_codes
    ]
    fl_times = [
        fastest_lap_time if d == fastest_lap_driver else pd.NA
        for d in driver_codes
    ]

    return pd.DataFrame({
        "round_number":      round_num,
        "event_name":        session.event["EventName"],
        "position":          pd.array(res["Position"], dtype="Int64"),
        "driver_number":     res["DriverNumber"].astype(str),
        "driver_code":       res["Abbreviation"],
        "full_name":         res["FullName"],
        "team_name":         res["TeamName"],
        "grid_position":     pd.array(res["GridPosition"], dtype="Int64"),
        "classified_status": res["Status"],
        "points":            res["Points"].astype(float),
        "fastest_lap_time":  fl_times,
        "fastest_lap_rank":  pd.array(fl_ranks, dtype="Int64"),
        "time":              res["Time"].astype(str),
    })


def extract_qualifying_results(session, round_num: int) -> pd.DataFrame:
    res = session.results.copy()
    return pd.DataFrame({
        "round_number": round_num,
        "event_name":   session.event["EventName"],
        "driver_code":  res["Abbreviation"],
        "team_name":    res["TeamName"],
        "q1_time":      res["Q1"].astype(str),
        "q2_time":      res["Q2"].astype(str),
        "q3_time":      res["Q3"].astype(str),
        "position":     pd.array(res["Position"], dtype="Int64"),
    })


def extract_fastest_lap_telemetry(session, round_num: int,
                                   logger: logging.Logger) -> pd.DataFrame | None:
    try:
        laps = session.laps
        valid = laps.pick_wo_box()
        try:
            valid = valid.pick_track_status("1")
        except Exception:
            pass  # some sessions don't have track status data

        if valid.empty:
            valid = laps.pick_wo_box()
        if valid.empty:
            logger.warning(f"  Round {round_num}: no valid laps for telemetry extraction")
            return None

        fastest = valid.pick_fastest()
        driver_code = fastest["Driver"]
        lap_time    = fastest["LapTime"]

        telemetry = fastest.get_telemetry()
        telemetry = telemetry.dropna(subset=["X", "Y"])

        if telemetry.empty:
            logger.warning(f"  Round {round_num}: telemetry empty after position filter")
            return None

        return pd.DataFrame({
            "time_ms":     (telemetry["Time"].dt.total_seconds() * 1000).astype("int64"),
            "x":           telemetry["X"].astype("float32"),
            "y":           telemetry["Y"].astype("float32"),
            "speed":       telemetry["Speed"].astype("float32"),
            "throttle":    telemetry["Throttle"].astype("float32"),
            "brake":       telemetry["Brake"].astype(bool),
            "gear":        telemetry["nGear"].astype("int8"),
            "rpm":         telemetry["RPM"].astype("float32"),
            "drs":         telemetry["DRS"].astype("int8"),
            "driver_code": driver_code,
            "lap_time_ms": int(lap_time.total_seconds() * 1000),
        })

    except Exception as e:
        logger.warning(f"  Round {round_num}: telemetry extraction failed — {e}")
        return None


def extract_weather_meta(session, round_num: int) -> dict:
    w = session.weather_data
    rainfall = bool(w["Rainfall"].any()) if not w.empty and "Rainfall" in w.columns else False

    def safe_mean(col):
        return float(w[col].mean()) if not w.empty and col in w.columns else None

    def safe_min(col):
        return float(w[col].min()) if not w.empty and col in w.columns else None

    def safe_max(col):
        return float(w[col].max()) if not w.empty and col in w.columns else None

    return {
        "round_number":       round_num,
        "event_name":         session.event["EventName"],
        "circuit_short_name": session.event.get("Location", ""),
        "country":            session.event.get("Country", ""),
        "location":           session.event.get("Location", ""),
        "session_date":       str(session.date.date()),
        "session_type":       "Race",
        "air_temp_avg":       safe_mean("AirTemp"),
        "air_temp_min":       safe_min("AirTemp"),
        "air_temp_max":       safe_max("AirTemp"),
        "track_temp_avg":     safe_mean("TrackTemp"),
        "track_temp_min":     safe_min("TrackTemp"),
        "track_temp_max":     safe_max("TrackTemp"),
        "humidity_avg":       safe_mean("Humidity"),
        "rainfall":           rainfall,
        "wind_speed_avg":     safe_mean("WindSpeed"),
        "weather_description": "Wet" if rainfall else "Dry",
    }


# ── Save ──────────────────────────────────────────────────────────────────────
def save_race_results(df: pd.DataFrame, round_num: int, slug: str):
    df.to_csv(RESULTS_DIR / f"race_{round_num:02d}_{slug}.csv", index=False)


def save_qualifying_results(df: pd.DataFrame, round_num: int, slug: str):
    df.to_csv(RESULTS_DIR / f"qualifying_{round_num:02d}_{slug}.csv", index=False)


def save_telemetry(df: pd.DataFrame, round_num: int, slug: str):
    path = TELEM_DIR / f"fastest_lap_{round_num:02d}_{slug}.parquet"
    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")


def save_meta(meta: dict, round_num: int, slug: str):
    path = META_DIR / f"meta_{round_num:02d}_{slug}.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)


# ── Per-round orchestration ───────────────────────────────────────────────────
def fetch_round(round_num: int, event_name: str, force: bool,
                logger: logging.Logger) -> bool:
    slug = event_name_to_slug(event_name)

    if not force and is_race_already_fetched(round_num, slug):
        logger.info(f"Round {round_num:02d} ({event_name}): already fetched — skipping.")
        return True

    logger.info(f"Round {round_num:02d} ({event_name}): fetching...")

    # Race session (telemetry + weather)
    race_session, ok = load_session_safe(SEASON_YEAR, round_num, "R", logger)
    if not ok:
        logger.warning(f"  Round {round_num}: race session unavailable — skipping round.")
        return False

    race_df = extract_race_results(race_session, round_num)
    save_race_results(race_df, round_num, slug)
    logger.info(f"  Round {round_num}: race results saved ({len(race_df)} drivers)")

    telem_df = extract_fastest_lap_telemetry(race_session, round_num, logger)
    if telem_df is not None:
        save_telemetry(telem_df, round_num, slug)
        logger.info(f"  Round {round_num}: telemetry saved ({len(telem_df)} rows, driver: {telem_df['driver_code'].iloc[0]})")
    else:
        logger.warning(f"  Round {round_num}: no telemetry saved")

    meta = extract_weather_meta(race_session, round_num)
    save_meta(meta, round_num, slug)
    logger.info(f"  Round {round_num}: weather meta saved")

    # Qualifying session (no telemetry needed)
    qual_session, ok = load_session_safe(SEASON_YEAR, round_num, "Q", logger)
    if ok:
        qual_df = extract_qualifying_results(qual_session, round_num)
        save_qualifying_results(qual_df, round_num, slug)
        logger.info(f"  Round {round_num}: qualifying results saved")
    else:
        logger.warning(f"  Round {round_num}: qualifying session unavailable")

    logger.info(f"Round {round_num:02d}: done.")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fetch 2025 F1 season data via FastF1")
    parser.add_argument(
        "--rounds", nargs=2, type=int, metavar=("START", "END"),
        help="Only fetch rounds START through END inclusive (e.g. --rounds 1 5)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-fetch data even if output files already exist",
    )
    args = parser.parse_args()

    logger = setup_logging()
    setup_directories()
    setup_fastf1_cache()

    schedule = RACE_SCHEDULE_2025
    if args.rounds:
        start, end = args.rounds
        schedule = [(n, name) for n, name in schedule if start <= n <= end]

    logger.info(f"Fetching {len(schedule)} round(s) for the 2025 F1 season...")
    logger.info(f"Output directory: {RAW_DIR}")

    success, failed = 0, []
    for round_num, event_name in schedule:
        ok = fetch_round(round_num, event_name, force=args.force, logger=logger)
        if ok:
            success += 1
        else:
            failed.append(round_num)

    logger.info(f"Fetch complete: {success}/{len(schedule)} rounds processed.")
    if failed:
        logger.warning(f"Failed rounds: {failed}")


if __name__ == "__main__":
    main()
