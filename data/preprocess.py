"""
preprocess.py
-------------
Reads raw fetched data and produces clean output files for the dashboard.
Must be run AFTER fetch_season_data.py.

Usage:
    python data/preprocess.py
    python data/preprocess.py --turning-point-round 23   # Qatar as turning point
    python data/preprocess.py --contenders NOR VER PIA
"""

import pickle
import pandas as pd
import numpy as np
import json
import logging
import argparse
import shutil
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TELEM_RAW_DIR = RAW_DIR / "telemetry"
TELEM_OUT_DIR = PROCESSED_DIR / "telemetry"

# ── Defaults (overridden by CLI args) ─────────────────────────────────────────
CONTENDERS     = ["NOR", "VER", "PIA"]
TURNING_POINT  = 14
TOTAL_ROUNDS   = 24
MC_SIMULATIONS = 50_000
MC_BATCH       = 10_000   # vectorised batch size for ML simulation

FULL_POINTS    = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]  # P1-P10

RAW_2024_DIR   = PROJECT_ROOT / "data" / "raw" / "2024"

SCENARIOS = [
    {
        "id":                   "hungary_ver",
        "name":                 "The Budapest Reversal",
        "round":                14,
        "description":          "Verstappen wins the Hungarian GP instead of Norris — a 30-point swing that fundamentally reshapes the title fight from mid-season.",
        "hook":                 "VER +30 pt swing",
        "changes":              {"VER": +23, "NOR": -7},   # VER: 2→25, NOR: 25→18
        "p_incident_driver":    "VER",
        "p_incident_condition": "win",
    },
    {
        "id":                   "mexico_ver",
        "name":                 "Mexico City Redemption",
        "round":                20,
        "description":          "Verstappen takes victory in Mexico City instead of Norris, cutting the gap to 26 points with 4 races remaining.",
        "hook":                 "VER +17 pt swing",
        "changes":              {"VER": +10, "NOR": -7},   # VER: 15→25, NOR: 25→18
        "p_incident_driver":    "VER",
        "p_incident_condition": "win",
    },
    {
        "id":                   "saopaulo_ver",
        "name":                 "Interlagos What-If",
        "round":                21,
        "description":          "Verstappen wins São Paulo from P19 on the grid. A chaotic race — and a 17-point swing that would have made Abu Dhabi a genuine decider.",
        "hook":                 "VER +17 pt swing",
        "changes":              {"VER": +10, "NOR": -7},   # VER: 15→25, NOR: 25→18
        "p_incident_driver":    "VER",
        "p_incident_condition": "win",
    },
    {
        "id":                   "qatar_kimi",
        "name":                 "The Kimi Moment",
        "round":                23,
        "description":          "Kimi Antonelli holds P4 and doesn't yield to Norris in Qatar. Just 2 points — but they cut the final margin from 5 to 3.",
        "hook":                 "2 pts · margin becomes 3",
        "changes":              {"NOR": -2},                # NOR: 12→10 (P4→P5)
        "p_incident_driver":    "ANT",
        "p_incident_condition": "beats_NOR",
    },
    {
        "id":                   "all_ver",
        "name":                 "Everything Goes Right",
        "round":                14,
        "description":          "All four incidents go Verstappen's way: Budapest, Mexico, São Paulo, and Qatar. The full picture of what it would have taken.",
        "hook":                 "All 4 incidents · VER champion",
        "changes":              {"VER": +23, "NOR": -7},   # Hungary base; rest applied cumulatively below
        "extra_changes": [
            {"round": 20, "changes": {"VER": +10, "NOR": -7}},
            {"round": 21, "changes": {"VER": +10, "NOR": -7}},
            {"round": 23, "changes": {"NOR": -2}},
        ],
        "p_incident_driver":    None,
        "p_incident_condition": "compound",
    },
]

NARRATIVE_PHASES = {
    **{r: "Season Start"      for r in range(1, 5)},
    **{r: "McLaren Dominance" for r in range(5, 10)},
    **{r: "Three-Way Battle"  for r in range(10, 14)},
    **{r: "Turning Point"     for r in range(14, 15)},
    **{r: "Finale"            for r in range(15, 25)},
}


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_all_race_results() -> pd.DataFrame:
    files = sorted((RAW_DIR / "results").glob("race_*.csv"))
    if not files:
        raise FileNotFoundError(
            "No race CSV files found in data/raw/results/. Run fetch_season_data.py first."
        )
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, dtype={"driver_number": str, "driver_code": str}))
        except Exception as e:
            logging.warning(f"Could not read {f.name}: {e}")
    return pd.concat(dfs, ignore_index=True).sort_values("round_number").reset_index(drop=True)


def load_all_qualifying_results() -> pd.DataFrame:
    files = sorted((RAW_DIR / "results").glob("qualifying_*.csv"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True).sort_values("round_number").reset_index(drop=True)


def load_all_meta() -> dict:
    """Returns {round_number: meta_dict}"""
    meta = {}
    for f in (RAW_DIR / "session_meta").glob("meta_*.json"):
        with open(f) as fp:
            d = json.load(fp)
        meta[int(d["round_number"])] = d
    return meta


# ── Season Progression ────────────────────────────────────────────────────────
def build_season_progression(race_results: pd.DataFrame) -> pd.DataFrame:
    """Cumulative points per driver per round."""
    completed_rounds = sorted(race_results["round_number"].unique())

    # Map driver → team (use last-seen team if driver changes)
    driver_team = (
        race_results.sort_values("round_number")
        .groupby("driver_code")["team_name"]
        .last()
        .to_dict()
    )
    all_drivers = race_results["driver_code"].unique()

    rows = []
    for driver in all_drivers:
        cumulative = 0.0
        for rnd in completed_rounds:
            rnd_mask = (race_results["round_number"] == rnd) & (race_results["driver_code"] == driver)
            rnd_row  = race_results[rnd_mask]

            if rnd_row.empty:
                race_pts = 0.0
                position = pd.NA
            else:
                race_pts = float(rnd_row["points"].iloc[0])
                position = rnd_row["position"].iloc[0]

            cumulative += race_pts
            event_name  = race_results[race_results["round_number"] == rnd]["event_name"].iloc[0]

            rows.append({
                "round_number":      rnd,
                "event_name":        event_name,
                "driver_code":       driver,
                "team_name":         driver_team.get(driver, ""),
                "race_points":       race_pts,
                "cumulative_points": cumulative,
                "position":          position,
                "narrative_phase":   NARRATIVE_PHASES.get(rnd, "Finale"),
            })

    df = pd.DataFrame(rows).sort_values(["driver_code", "round_number"]).reset_index(drop=True)
    df["position"] = pd.array(df["position"], dtype="Int64")
    return df


# ── Race Cards ────────────────────────────────────────────────────────────────
def build_race_cards(race_results: pd.DataFrame,
                     qual_results: pd.DataFrame,
                     meta_dict: dict) -> pd.DataFrame:
    """One row per race: top 3, weather, pole, fastest lap, narrative phase."""
    cards = []

    for rnd in sorted(race_results["round_number"].unique()):
        rnd_race = race_results[race_results["round_number"] == rnd].copy()
        rnd_race = rnd_race.sort_values("position")

        def get_driver(pos):
            row = rnd_race[rnd_race["position"] == pos]
            return row["driver_code"].iloc[0] if not row.empty else None

        def get_team(pos):
            row = rnd_race[rnd_race["position"] == pos]
            return row["team_name"].iloc[0] if not row.empty else None

        fl_row    = rnd_race[rnd_race["fastest_lap_rank"] == 1]
        fl_driver = fl_row["driver_code"].iloc[0] if not fl_row.empty else None
        fl_time   = fl_row["fastest_lap_time"].iloc[0] if not fl_row.empty else None

        pole_driver = pole_time = None
        if not qual_results.empty:
            rnd_qual = qual_results[qual_results["round_number"] == rnd]
            pole_row = rnd_qual[rnd_qual["position"] == 1]
            if not pole_row.empty:
                pole_driver = pole_row["driver_code"].iloc[0]
                pole_time   = pole_row["q3_time"].iloc[0]

        meta = meta_dict.get(rnd, {})

        cards.append({
            "round_number":        rnd,
            "event_name":          rnd_race["event_name"].iloc[0],
            "circuit_short_name":  meta.get("circuit_short_name", ""),
            "country":             meta.get("country", ""),
            "session_date":        meta.get("session_date", ""),
            "p1_driver":           get_driver(1),
            "p2_driver":           get_driver(2),
            "p3_driver":           get_driver(3),
            "p1_team":             get_team(1),
            "p2_team":             get_team(2),
            "p3_team":             get_team(3),
            "fastest_lap_driver":  fl_driver,
            "fastest_lap_time":    fl_time,
            "pole_driver":         pole_driver,
            "pole_time":           pole_time,
            "track_temp_avg":      meta.get("track_temp_avg"),
            "air_temp_avg":        meta.get("air_temp_avg"),
            "rainfall":            bool(meta.get("rainfall", False)),
            "weather_description": meta.get("weather_description", "Dry"),
            "narrative_phase":     NARRATIVE_PHASES.get(rnd, "Finale"),
            "total_laps":          int(rnd_race["position"].notna().sum()),
        })

    return pd.DataFrame(cards)


# ── Championship Probability (Monte Carlo) ────────────────────────────────────
def compute_championship_probabilities(season_prog: pd.DataFrame) -> pd.DataFrame:
    """
    After each completed round, compute P(win championship) for each contender
    via Monte Carlo simulation over the remaining races.

    Uses Elo-style form ratings updated race-by-race to weight win probabilities.
    """
    completed_rounds = sorted(season_prog["round_number"].unique())
    form = {d: 1000.0 for d in CONTENDERS}
    K    = 20.0
    rows = []

    for rnd in completed_rounds:
        # Update form ratings based on this round's finish order
        rnd_data = season_prog[
            (season_prog["round_number"] == rnd) &
            (season_prog["driver_code"].isin(CONTENDERS))
        ].set_index("driver_code")

        positions = {}
        for driver in CONTENDERS:
            if driver in rnd_data.index:
                pos = rnd_data.loc[driver, "position"]
                positions[driver] = int(pos) if pd.notna(pos) else 20
            else:
                positions[driver] = 20

        total_form = sum(form.values())
        for driver in CONTENDERS:
            actual   = 1.0 if positions[driver] == 1 else (0.5 if positions[driver] <= 3 else 0.0)
            expected = form[driver] / total_form
            form[driver] = max(1.0, form[driver] + K * (actual - expected))

        # Current cumulative points for contenders
        contender_pts = {}
        for driver in CONTENDERS:
            row = season_prog[
                (season_prog["round_number"] == rnd) &
                (season_prog["driver_code"] == driver)
            ]
            contender_pts[driver] = float(row["cumulative_points"].iloc[0]) if not row.empty else 0.0

        leader_pts       = max(contender_pts.values())
        races_remaining  = TOTAL_ROUNDS - rnd

        # Win probabilities via normalized form ratings
        form_vals  = np.array([form[d] for d in CONTENDERS], dtype=float)
        win_probs  = form_vals / form_vals.sum()

        # Monte Carlo
        win_counts    = {d: 0 for d in CONTENDERS}
        podium_counts = {d: 0 for d in CONTENDERS}
        rng = np.random.default_rng(seed=42 + rnd)

        for _ in range(MC_SIMULATIONS):
            sim_pts = contender_pts.copy()

            for _ in range(races_remaining):
                order = rng.choice(len(CONTENDERS), size=len(CONTENDERS), replace=False, p=win_probs)
                pts_awards = [25.0, 18.0, 15.0]
                for rank, driver_idx in enumerate(order):
                    driver = CONTENDERS[driver_idx]
                    sim_pts[driver] += pts_awards[rank] if rank < len(pts_awards) else 0.0
                # 20% chance fastest lap point goes to the race winner
                if rng.random() < 0.2:
                    sim_pts[CONTENDERS[order[0]]] += 1.0

            champ      = max(sim_pts, key=sim_pts.get)
            sorted_pts = sorted(sim_pts.values(), reverse=True)
            win_counts[champ] += 1
            for d in CONTENDERS:
                if sim_pts[d] >= sorted_pts[min(2, len(sorted_pts) - 1)]:
                    podium_counts[d] += 1

        event_name = season_prog[season_prog["round_number"] == rnd]["event_name"].iloc[0]
        for driver in CONTENDERS:
            rows.append({
                "round_number":             rnd,
                "event_name":               event_name,
                "driver_code":              driver,
                "points_after_race":        contender_pts[driver],
                "points_gap_to_leader":     contender_pts[driver] - leader_pts,
                "races_remaining":          races_remaining,
                "max_points_available":     races_remaining * 26,
                "championship_probability": win_counts[driver] / MC_SIMULATIONS,
                "win_probability":          win_counts[driver] / MC_SIMULATIONS,
                "podium_probability":       podium_counts[driver] / MC_SIMULATIONS,
            })

    return pd.DataFrame(rows)


# ── ML Championship Predictor ─────────────────────────────────────────────────
def load_2024_data() -> tuple:
    """Load 2024 race results from data/raw/2024/."""
    race_files = sorted(RAW_2024_DIR.glob("race_*.csv"))
    if not race_files:
        raise FileNotFoundError("No 2024 race files. Run fetch_2024_data.py first.")

    race_2024 = pd.concat(
        [pd.read_csv(f) for f in race_files], ignore_index=True
    ).sort_values("round_number").reset_index(drop=True)

    return race_2024


def build_training_features(race_2024: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-driver-race feature rows from 2024 data.

    Features: grid_position, rolling_pts_3, driver_season_avg_grid
    Target (finish_zone): 0=win  1=podium(P2-P3)  2=points(P4-P10)  3=no-points
    """
    rows = []
    for rnd in sorted(race_2024["round_number"].unique()):
        rnd_race = race_2024[race_2024["round_number"] == rnd]

        for _, row in rnd_race.iterrows():
            driver = row["driver_code"]
            pts    = float(row["points"])

            grid_pos = row["grid_position"]
            grid_pos = float(grid_pos) if pd.notna(grid_pos) else 10.0

            # Finish zone target (>= 25 handles fastest-lap bonus giving 26 pts)
            if pts >= 25:
                zone = 0
            elif pts >= 15:
                zone = 1
            elif pts > 0:
                zone = 2
            else:
                zone = 3

            past = race_2024[
                (race_2024["round_number"] < rnd) &
                (race_2024["driver_code"]  == driver)
            ].sort_values("round_number")

            rolling_pts_3   = float(past["points"].tail(3).mean()) if not past.empty else 0.0
            past_grid       = past["grid_position"].dropna()
            season_avg_grid = float(past_grid.mean()) if not past_grid.empty else grid_pos

            rows.append({
                "grid_position":          grid_pos,
                "rolling_pts_3":          rolling_pts_3,
                "driver_season_avg_grid": season_avg_grid,
                "finish_zone":            zone,
            })

    return pd.DataFrame(rows)


def train_race_model(features_df: pd.DataFrame):
    """Train a GradientBoostingClassifier on 2024 feature rows."""
    from sklearn.ensemble import GradientBoostingClassifier

    X = features_df[["grid_position", "rolling_pts_3", "driver_season_avg_grid"]].values
    y = features_df["finish_zone"].values

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X, y)
    return model


def compute_championship_probabilities_ml(
    season_prog: pd.DataFrame,
    race_results: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """
    ML-based championship probabilities via full 20-driver vectorised Monte Carlo.

    Win probabilities come from the GBM model; remaining races simulate all drivers
    and award full F1 points (P1-P10).  Output schema matches the Elo version so the
    dashboard reads predictor_probs.csv unchanged.
    """
    completed_rounds = sorted(race_results["round_number"].unique())
    all_drivers      = sorted(race_results["driver_code"].unique())
    n_drivers        = len(all_drivers)
    d_to_idx         = {d: i for i, d in enumerate(all_drivers)}
    c_idx            = np.array(
        [d_to_idx[d] for d in CONTENDERS if d in d_to_idx], dtype=np.intp
    )

    # Points lookup by 0-based finishing position
    pts_by_pos = np.zeros(n_drivers, dtype=np.float32)
    for i, p in enumerate(FULL_POINTS):
        if i < n_drivers:
            pts_by_pos[i] = float(p)

    win_class_idx = int(np.where(model.classes_ == 0)[0][0]) if 0 in model.classes_ else 0

    rows = []

    for rnd in completed_rounds:
        races_remaining = TOTAL_ROUNDS - rnd
        rnd_prog        = season_prog[season_prog["round_number"] == rnd].set_index("driver_code")
        event_name      = season_prog[season_prog["round_number"] == rnd]["event_name"].iloc[0]

        curr_pts = np.array(
            [float(rnd_prog.loc[d, "cumulative_points"]) if d in rnd_prog.index else 0.0
             for d in all_drivers],
            dtype=np.float32,
        )
        contender_pts = {d: float(curr_pts[d_to_idx[d]]) for d in CONTENDERS if d in d_to_idx}
        leader_pts    = max(contender_pts.values()) if contender_pts else 0.0

        if races_remaining == 0:
            champ = max(contender_pts, key=contender_pts.get)
            top3  = sorted(contender_pts.values(), reverse=True)[:3]
            for driver in CONTENDERS:
                cp = contender_pts.get(driver, 0.0)
                rows.append({
                    "round_number":             rnd,
                    "event_name":               event_name,
                    "driver_code":              driver,
                    "points_after_race":        cp,
                    "points_gap_to_leader":     cp - leader_pts,
                    "races_remaining":          0,
                    "max_points_available":     0,
                    "championship_probability": 1.0 if driver == champ else 0.0,
                    "win_probability":          1.0 if driver == champ else 0.0,
                    "podium_probability":       1.0 if cp >= top3[-1] else 0.0,
                })
            continue

        # Build ML features for all drivers at this round
        feats = []
        for driver in all_drivers:
            past = race_results[
                (race_results["round_number"] <= rnd) &
                (race_results["driver_code"]  == driver)
            ].sort_values("round_number")

            rolling_pts_3   = float(past["points"].tail(3).mean()) if not past.empty else 0.0
            past_grid       = past["grid_position"].dropna()
            season_avg_grid = float(past_grid.mean()) if not past_grid.empty else 10.0

            # Use season avg grid as proxy for unknown future grid position
            feats.append([season_avg_grid, rolling_pts_3, season_avg_grid])

        X_sim      = np.array(feats, dtype=np.float32)              # (n_drivers, 3)
        zone_probs = model.predict_proba(X_sim)                     # (n_drivers, n_classes)
        raw_probs  = zone_probs[:, win_class_idx].astype(np.float32)
        total      = raw_probs.sum()
        win_probs  = raw_probs / total if total > 1e-9 else np.full(n_drivers, 1.0 / n_drivers, dtype=np.float32)
        log_p      = np.log(win_probs + 1e-10)                      # (n_drivers,)

        # Vectorised batch Monte Carlo (Gumbel-max trick for weighted permutations)
        win_counts = np.zeros(len(CONTENDERS), dtype=np.int64)
        rng        = np.random.default_rng(seed=42 + rnd)
        n_batches  = MC_SIMULATIONS // MC_BATCH

        for _ in range(n_batches):
            # (MC_BATCH, races_remaining, n_drivers)
            gumbel      = rng.gumbel(size=(MC_BATCH, races_remaining, n_drivers)).astype(np.float32)
            scores      = log_p + gumbel
            race_orders = np.argsort(-scores, axis=2).astype(np.int32)   # finish order per race
            ranks       = np.argsort(race_orders, axis=2).astype(np.int32)  # position of each driver

            pts_awarded = pts_by_pos[ranks]                          # (MC_BATCH, races, n_drivers)
            sim_extra   = pts_awarded.sum(axis=1)                    # (MC_BATCH, n_drivers)

            # Fastest lap: P1 winner gets +1 with 20% probability
            p1_idx  = race_orders[:, :, 0]                           # (MC_BATCH, races)
            fl_mask = (rng.random((MC_BATCH, races_remaining)) < 0.2).astype(np.float32)
            for r in range(races_remaining):
                sim_extra[np.arange(MC_BATCH), p1_idx[:, r]] += fl_mask[:, r]

            final_pts       = curr_pts + sim_extra                   # (MC_BATCH, n_drivers)
            contender_final = final_pts[:, c_idx]                    # (MC_BATCH, n_contenders)
            champ_local_idx = np.argmax(contender_final, axis=1)     # (MC_BATCH,)
            win_counts     += np.bincount(champ_local_idx, minlength=len(CONTENDERS))

        for i, driver in enumerate(CONTENDERS):
            cp = contender_pts.get(driver, 0.0)
            rows.append({
                "round_number":             rnd,
                "event_name":               event_name,
                "driver_code":              driver,
                "points_after_race":        cp,
                "points_gap_to_leader":     cp - leader_pts,
                "races_remaining":          races_remaining,
                "max_points_available":     races_remaining * 26,
                "championship_probability": int(win_counts[i]) / MC_SIMULATIONS,
                "win_probability":          int(win_counts[i]) / MC_SIMULATIONS,
                "podium_probability":       0.0,
            })

    return pd.DataFrame(rows)


# ── Alternate Reality ─────────────────────────────────────────────────────────
def build_alternate_reality(season_prog: pd.DataFrame, turning_point: int) -> pd.DataFrame:
    """
    Counterfactual: at `turning_point`, VER scores 25 pts (wins) instead of actual.
    Subsequent rounds keep the same actual race increments on the altered base.
    """
    contender_data = (
        season_prog[season_prog["driver_code"].isin(CONTENDERS)]
        .copy()
        .sort_values(["driver_code", "round_number"])
    )

    rows = []
    for driver in CONTENDERS:
        driver_rows    = contender_data[contender_data["driver_code"] == driver]
        alt_cumulative = 0.0

        for _, row in driver_rows.iterrows():
            rnd        = int(row["round_number"])
            actual_pts = float(row["race_points"])

            if rnd < turning_point:
                alt_pts = actual_pts
                is_alt  = False
            elif rnd == turning_point:
                is_alt  = True
                alt_pts = 25.0 if driver == "VER" else actual_pts
            else:
                alt_pts = actual_pts
                is_alt  = True

            alt_cumulative += alt_pts
            rows.append({
                "round_number":             rnd,
                "event_name":               row["event_name"],
                "driver_code":              driver,
                "actual_cumulative_points": float(row["cumulative_points"]),
                "alt_cumulative_points":    alt_cumulative,
                "actual_race_points":       actual_pts,
                "alt_race_points":          alt_pts,
                "scenario_name":            f"Round {turning_point} VER Win",
                "is_alt_race":              is_alt,
                "delta":                    alt_cumulative - float(row["cumulative_points"]),
            })

    return pd.DataFrame(rows)


# ── Alternate Reality Scenarios (VER-centric) ─────────────────────────────────
def _run_alt_mc(curr_pts_map: dict, race_results: pd.DataFrame,
                model, rnd: int, rng_seed: int) -> dict:
    if model is None:
        return {d: 0.0 for d in CONTENDERS}
    """Run a single-round ML MC with custom starting points. Returns {driver: prob}."""
    all_drivers = sorted(race_results["driver_code"].unique())
    n_drivers   = len(all_drivers)
    d_to_idx    = {d: i for i, d in enumerate(all_drivers)}
    c_idx       = np.array([d_to_idx[d] for d in CONTENDERS if d in d_to_idx], dtype=np.intp)

    pts_by_pos = np.zeros(n_drivers, dtype=np.float32)
    for i, p in enumerate(FULL_POINTS):
        if i < n_drivers:
            pts_by_pos[i] = float(p)

    win_class_idx = int(np.where(model.classes_ == 0)[0][0]) if 0 in model.classes_ else 0
    races_remaining = TOTAL_ROUNDS - rnd

    curr_pts = np.array(
        [float(curr_pts_map.get(d, 0.0)) for d in all_drivers],
        dtype=np.float32,
    )

    if races_remaining == 0:
        contender_pts = {d: float(curr_pts[d_to_idx[d]]) for d in CONTENDERS if d in d_to_idx}
        champ = max(contender_pts, key=contender_pts.get)
        return {d: (1.0 if d == champ else 0.0) for d in CONTENDERS}

    # ML features for all drivers
    feats = []
    for driver in all_drivers:
        past = race_results[
            (race_results["round_number"] <= rnd) &
            (race_results["driver_code"]  == driver)
        ].sort_values("round_number")
        rolling_pts_3   = float(past["points"].tail(3).mean()) if not past.empty else 0.0
        past_grid       = past["grid_position"].dropna()
        season_avg_grid = float(past_grid.mean()) if not past_grid.empty else 10.0
        feats.append([season_avg_grid, rolling_pts_3, season_avg_grid])

    X_sim     = np.array(feats, dtype=np.float32)
    raw_probs = model.predict_proba(X_sim)[:, win_class_idx].astype(np.float32)
    total     = raw_probs.sum()
    win_probs = raw_probs / total if total > 1e-9 else np.full(n_drivers, 1.0 / n_drivers, dtype=np.float32)
    log_p     = np.log(win_probs + 1e-10)

    ALT_SIMS  = 20_000
    ALT_BATCH = 10_000
    win_counts = np.zeros(len(CONTENDERS), dtype=np.int64)
    rng = np.random.default_rng(seed=rng_seed)

    for _ in range(ALT_SIMS // ALT_BATCH):
        gumbel      = rng.gumbel(size=(ALT_BATCH, races_remaining, n_drivers)).astype(np.float32)
        scores      = log_p + gumbel
        race_orders = np.argsort(-scores, axis=2).astype(np.int32)
        ranks       = np.argsort(race_orders, axis=2).astype(np.int32)
        pts_awarded = pts_by_pos[ranks]
        sim_extra   = pts_awarded.sum(axis=1)
        p1_idx      = race_orders[:, :, 0]
        fl_mask     = (rng.random((ALT_BATCH, races_remaining)) < 0.2).astype(np.float32)
        for r in range(races_remaining):
            sim_extra[np.arange(ALT_BATCH), p1_idx[:, r]] += fl_mask[:, r]
        final_pts       = curr_pts + sim_extra
        champ_local_idx = np.argmax(final_pts[:, c_idx], axis=1)
        win_counts     += np.bincount(champ_local_idx, minlength=len(CONTENDERS))

    return {d: int(win_counts[i]) / ALT_SIMS for i, d in enumerate(CONTENDERS)}


def compute_race_win_prob(
    model,
    race_results: pd.DataFrame,
    rnd: int,
    condition: str,
    primary_driver: str,
    n_sims: int = 20_000,
    rng_seed: int = 42,
) -> float:
    """
    Estimate P(incident) for a single race round using Gumbel-max MC.

    condition == "win":       P(primary_driver finishes P1 in race rnd)
    condition == "beats_NOR": P(primary_driver finishes ahead of NOR in race rnd)
    """
    if model is None or not primary_driver:
        return 0.0

    all_drivers = sorted(race_results["driver_code"].unique())
    n_drivers   = len(all_drivers)
    d_to_idx    = {d: i for i, d in enumerate(all_drivers)}

    if primary_driver not in d_to_idx:
        return 0.0

    driver_idx    = d_to_idx[primary_driver]
    win_class_idx = int(np.where(model.classes_ == 0)[0][0]) if 0 in model.classes_ else 0

    feats = []
    for driver in all_drivers:
        past = race_results[
            (race_results["round_number"] <= rnd) &
            (race_results["driver_code"]  == driver)
        ].sort_values("round_number")
        rolling_pts_3   = float(past["points"].tail(3).mean()) if not past.empty else 0.0
        past_grid       = past["grid_position"].dropna()
        season_avg_grid = float(past_grid.mean()) if not past_grid.empty else 10.0
        feats.append([season_avg_grid, rolling_pts_3, season_avg_grid])

    X_sim     = np.array(feats, dtype=np.float32)
    raw_probs = model.predict_proba(X_sim)[:, win_class_idx].astype(np.float32)
    total     = raw_probs.sum()
    win_probs = raw_probs / total if total > 1e-9 else np.full(n_drivers, 1.0 / n_drivers, dtype=np.float32)
    log_p     = np.log(win_probs + 1e-10)

    rng           = np.random.default_rng(seed=rng_seed)
    BATCH         = 10_000
    success_count = 0

    for _ in range(max(1, n_sims // BATCH)):
        gumbel      = rng.gumbel(size=(BATCH, n_drivers)).astype(np.float32)
        scores      = log_p + gumbel
        race_orders = np.argsort(-scores, axis=1)

        if condition == "win":
            success_count += int(np.sum(race_orders[:, 0] == driver_idx))
        elif condition == "beats_NOR":
            nor_idx = d_to_idx.get("NOR", -1)
            if nor_idx == -1:
                return 0.0
            ranks = np.argsort(race_orders, axis=1)
            success_count += int(np.sum(ranks[:, driver_idx] < ranks[:, nor_idx]))

    return success_count / n_sims


def compute_alt_scenario_probs(
    scenario: dict,
    season_prog: pd.DataFrame,
    race_results: pd.DataFrame,
    model,
    actual_probs_df: pd.DataFrame,
    p_incident: float = 0.0,
) -> pd.DataFrame:
    """
    For one scenario: compute alt cumulative points + alt ML championship probabilities
    for NOR/VER/PIA across all 24 rounds.
    """
    turning_point = scenario["round"]
    base_changes  = scenario["changes"]
    extra_changes = scenario.get("extra_changes", [])   # for combined scenario

    # Build a lookup of all extra turning points → changes
    all_changes: dict[int, dict] = {turning_point: base_changes}
    for ec in extra_changes:
        rnd = ec["round"]
        all_changes[rnd] = ec["changes"]

    contender_data = (
        season_prog[season_prog["driver_code"].isin(CONTENDERS)]
        .copy()
        .sort_values(["driver_code", "round_number"])
    )

    # Walk through rounds building alt cumulative points per driver
    alt_cumulative: dict[str, float] = {d: 0.0 for d in CONTENDERS}
    rows = []

    for rnd in sorted(season_prog["round_number"].unique()):
        rnd_changes = all_changes.get(rnd, {})

        for driver in CONTENDERS:
            dr = contender_data[
                (contender_data["driver_code"] == driver) &
                (contender_data["round_number"] == rnd)
            ]
            if dr.empty:
                continue
            row        = dr.iloc[0]
            actual_pts = float(row["race_points"])
            actual_cum = float(row["cumulative_points"])

            alt_race_pts = actual_pts + rnd_changes.get(driver, 0.0)
            alt_cumulative[driver] += alt_race_pts

        # Actual championship probs from predictor_probs
        act_prob_rnd = actual_probs_df[actual_probs_df["round_number"] == rnd]

        # Alt probabilities: only recompute from earliest turning point onward
        if rnd >= turning_point:
            alt_probs = _run_alt_mc(
                curr_pts_map={d: alt_cumulative[d] for d in CONTENDERS},
                race_results=race_results,
                model=model,
                rnd=rnd,
                rng_seed=999 + rnd + hash(scenario["id"]) % 1000,
            )
        else:
            alt_probs = {
                d: float(act_prob_rnd[act_prob_rnd["driver_code"] == d]["championship_probability"].iloc[0])
                if not act_prob_rnd[act_prob_rnd["driver_code"] == d].empty else 0.0
                for d in CONTENDERS
            }

        for driver in CONTENDERS:
            dr = contender_data[
                (contender_data["driver_code"] == driver) &
                (contender_data["round_number"] == rnd)
            ]
            actual_cum = float(dr["cumulative_points"].iloc[0]) if not dr.empty else 0.0
            act_prob   = float(act_prob_rnd[act_prob_rnd["driver_code"] == driver]["championship_probability"].iloc[0]) \
                         if not act_prob_rnd[act_prob_rnd["driver_code"] == driver].empty else 0.0

            rows.append({
                "scenario_id":               scenario["id"],
                "scenario_name":             scenario["name"],
                "scenario_description":      scenario["description"],
                "scenario_hook":             scenario["hook"],
                "turning_point":             turning_point,
                "round_number":              rnd,
                "driver_code":               driver,
                "actual_cumulative_points":  actual_cum,
                "alt_cumulative_points":     alt_cumulative[driver],
                "delta":                     alt_cumulative[driver] - actual_cum,
                "actual_championship_prob":  act_prob,
                "alt_championship_prob":     alt_probs.get(driver, 0.0),
                "p_incident":                p_incident,
            })

    return pd.DataFrame(rows)


def build_all_alternate_scenarios(
    season_prog: pd.DataFrame,
    race_results: pd.DataFrame,
    model,
    actual_probs_df: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Run all VER-centric scenarios and return a combined DataFrame."""
    individual_dfs: list[pd.DataFrame] = []
    individual_p:   dict[str, float]   = {}

    for scenario in SCENARIOS:
        if scenario.get("p_incident_condition") == "compound":
            continue  # handled after individual scenarios

        logger.info(f"  Computing scenario: {scenario['name']}...")

        # Compute individual incident probability
        p_inc = compute_race_win_prob(
            model, race_results,
            rnd=scenario["round"],
            condition=scenario["p_incident_condition"],
            primary_driver=scenario["p_incident_driver"] or "",
            n_sims=20_000,
            rng_seed=42 + hash(scenario["id"]) % 997,
        )
        individual_p[scenario["id"]] = p_inc
        logger.info(f"    p_incident = {p_inc:.4f}")

        df = compute_alt_scenario_probs(
            scenario, season_prog, race_results, model, actual_probs_df,
            p_incident=p_inc,
        )
        individual_dfs.append(df)
        logger.info(f"    → {len(df)} rows")

    # Compound probability = product of all 4 individual incident probabilities
    compound_p = 1.0
    for p in individual_p.values():
        compound_p *= p
    logger.info(f"  Compound probability (all 4 incidents): {compound_p:.6f}  ({compound_p*100:.2f}%)")

    all_ver = next(s for s in SCENARIOS if s["id"] == "all_ver")
    logger.info(f"  Computing scenario: {all_ver['name']}...")
    all_ver_df = compute_alt_scenario_probs(
        all_ver, season_prog, race_results, model, actual_probs_df,
        p_incident=compound_p,
    )
    logger.info(f"    → {len(all_ver_df)} rows")

    return pd.concat(individual_dfs + [all_ver_df], ignore_index=True)


# ── Telemetry passthrough ─────────────────────────────────────────────────────
def copy_telemetry_to_processed(logger: logging.Logger):
    """Rename fastest_lap_NN_Slug.parquet → race_NN.parquet in processed/telemetry/."""
    TELEM_OUT_DIR.mkdir(parents=True, exist_ok=True)

    for raw_file in sorted(TELEM_RAW_DIR.glob("fastest_lap_*.parquet")):
        # fastest_lap_01_Australian_GP.parquet → extract '01'
        parts     = raw_file.stem.split("_")
        round_num = int(parts[2])
        out_path  = TELEM_OUT_DIR / f"race_{round_num:02d}.parquet"

        if out_path.exists():
            logger.info(f"  Telemetry race_{round_num:02d}: already exists — skipping.")
            continue

        shutil.copy2(raw_file, out_path)
        logger.info(f"  Telemetry race_{round_num:02d}: copied.")


# ── Driver Images ────────────────────────────────────────────────────────────
def build_driver_images(logger: logging.Logger):
    """Extract HeadshotUrl from FastF1 cache (Round 1, already on disk) → driver_images.csv."""
    out_path = PROCESSED_DIR / "driver_images.csv"
    try:
        import fastf1
        fastf1.Cache.enable_cache(str(PROJECT_ROOT / "cache"))  # must be str, not Path
        session = fastf1.get_session(2025, 1, "R")
        session.load(telemetry=False, weather=False, messages=False)
        df = session.results[["Abbreviation", "HeadshotUrl", "FullName", "TeamName"]].copy()
        df = df.rename(columns={
            "Abbreviation": "driver_code",
            "HeadshotUrl":  "headshot_url",
            "FullName":     "full_name",
            "TeamName":     "team_name",
        })
        df = df.reset_index(drop=True)
        df.to_csv(out_path, index=False)
        logger.info(f"  → driver_images.csv: {len(df)} drivers saved.")
    except Exception as e:
        logger.warning(f"Could not build driver_images.csv: {e}")
        pd.DataFrame(columns=["driver_code", "headshot_url", "full_name", "team_name"]).to_csv(
            out_path, index=False
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Preprocess 2025 F1 season data")
    parser.add_argument(
        "--turning-point-round", type=int, default=14,
        help="Round number for the alternate reality turning point (default: 14)",
    )
    parser.add_argument(
        "--contenders", nargs="+", default=["NOR", "VER", "PIA"],
        help="Driver codes for contender analysis (default: NOR VER PIA)",
    )
    args = parser.parse_args()

    global CONTENDERS, TURNING_POINT
    CONTENDERS    = args.contenders
    TURNING_POINT = args.turning_point_round

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger(__name__)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading raw data...")
    race_results = load_all_race_results()
    qual_results = load_all_qualifying_results()
    meta_dict    = load_all_meta()

    completed = race_results["round_number"].nunique()
    log.info(f"Found {completed} completed race(s).")

    log.info("Building season_progression.csv ...")
    season_prog = build_season_progression(race_results)
    season_prog.to_csv(PROCESSED_DIR / "season_progression.csv", index=False)
    log.info(f"  → {len(season_prog)} rows")

    log.info("Building contender_progression.csv ...")
    contender_prog = season_prog[season_prog["driver_code"].isin(CONTENDERS)].copy()
    contender_prog.to_csv(PROCESSED_DIR / "contender_progression.csv", index=False)
    log.info(f"  → {len(contender_prog)} rows")

    log.info("Building race_cards.csv ...")
    race_cards = build_race_cards(race_results, qual_results, meta_dict)
    race_cards.to_csv(PROCESSED_DIR / "race_cards.csv", index=False)
    log.info(f"  → {len(race_cards)} rows")

    # Championship probabilities — ML model if 2024 data exists, else Elo fallback
    ml_model = None
    if RAW_2024_DIR.exists() and any(RAW_2024_DIR.glob("race_*.csv")):
        try:
            log.info("Loading 2024 data for ML predictor...")
            race_2024    = load_2024_data()
            log.info(f"  → {len(race_2024)} driver-race rows from 2024")
            features_df  = build_training_features(race_2024)
            zone_dist    = features_df["finish_zone"].value_counts().sort_index().to_dict()
            log.info(f"  → Training rows: {len(features_df)}  zone distribution: {zone_dist}")
            ml_model     = train_race_model(features_df)
            model_path   = PROCESSED_DIR / "race_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(ml_model, f)
            log.info(f"  → GBM model trained and saved to {model_path.name}")
        except Exception as e:
            log.warning(f"ML model training failed ({e}) — falling back to Elo predictor")
            ml_model = None
    else:
        log.warning(
            "2024 data not found in data/raw/2024/ — "
            "run fetch_2024_data.py to enable ML predictor. Using Elo fallback."
        )

    if ml_model is not None:
        log.info(
            f"Building predictor_probs.csv "
            f"(ML + full-field Monte Carlo, {MC_SIMULATIONS:,} sims/round)..."
        )
        predictor_probs = compute_championship_probabilities_ml(
            season_prog, race_results, ml_model
        )
    else:
        log.info(f"Building predictor_probs.csv (Elo Monte Carlo, {MC_SIMULATIONS:,} sims/round)...")
        predictor_probs = compute_championship_probabilities(season_prog)

    predictor_probs.to_csv(PROCESSED_DIR / "predictor_probs.csv", index=False)
    log.info(f"  → {len(predictor_probs)} rows")

    log.info("Building alternate_scenarios.csv (VER-centric what-if stories)...")
    if ml_model is not None:
        alt_scenarios = build_all_alternate_scenarios(
            season_prog, race_results, ml_model, predictor_probs, log
        )
    else:
        log.warning("No ML model — skipping alternate scenario probabilities, using points-only.")
        alt_scenarios = build_all_alternate_scenarios(
            season_prog, race_results, None, predictor_probs, log
        )
    alt_scenarios.to_csv(PROCESSED_DIR / "alternate_scenarios.csv", index=False)
    log.info(f"  → {len(alt_scenarios)} rows")

    log.info("Copying telemetry to processed/ ...")
    copy_telemetry_to_processed(log)

    log.info("Building driver_images.csv ...")
    build_driver_images(log)

    log.info(f"Preprocessing complete. Output: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
