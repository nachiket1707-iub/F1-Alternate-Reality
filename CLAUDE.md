# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the dashboard
streamlit run dashboard/app.py

# Data pipeline (run in order)
python data/fetch_season_data.py                  # fetch all 24 rounds (~9 min first run, instant after cache)
python data/fetch_season_data.py --rounds 1 5     # fetch specific rounds only
python data/fetch_season_data.py --force          # re-fetch ignoring existing files
python data/preprocess.py                         # build processed files (~8 min due to Monte Carlo)
python data/preprocess.py --turning-point-round 23 --contenders NOR VER PIA
```

## Architecture

The project is a **strict two-phase pipeline**: data must be fully preprocessed before the dashboard starts. The dashboard never calls FastF1 or touches `data/raw/`.

```
data/fetch_season_data.py  →  data/raw/          →  data/preprocess.py  →  data/processed/  →  dashboard/app.py
      (FastF1 API)            (CSV + Parquet)         (pandas + MC)         (CSV + Parquet)      (Streamlit, read-only)
```

### Data layer (`data/`)

**`fetch_season_data.py`** — Fetches 2025 F1 season data (24 races) via FastF1 3.8.x. One key quirk: `session.results` in FastF1 3.8.x does **not** contain `FastestLapTime`/`FastestLapRank` columns — these are derived from `session.laps.pick_fastest()` inside `extract_race_results()`. FastF1 cache lives in `cache/` and must be enabled before any `get_session()` call.

**`preprocess.py`** — Reads all raw files, builds 5 processed CSVs + 24 telemetry parquets. The Monte Carlo step (`compute_championship_probabilities`) runs 50,000 simulations × 24 rounds using Elo-weighted win probabilities — takes ~8 minutes in pure Python. The alternate reality counterfactual (Round 14, VER wins instead of actual) is hardcoded as `--turning-point-round 14`.

### Processed file schemas (what the dashboard reads)

| File | Rows | Key columns |
|---|---|---|
| `season_progression.csv` | 504 | `round_number, driver_code, cumulative_points, narrative_phase` |
| `contender_progression.csv` | 72 | Same, filtered to NOR/VER/PIA only |
| `race_cards.csv` | 24 | `p1/p2/p3_driver, pole_driver, fastest_lap_time, weather_description, narrative_phase` |
| `predictor_probs.csv` | 72 | `championship_probability` sums to 1.0 per round |
| `alternate_reality.csv` | 72 | `actual_cumulative_points, alt_cumulative_points, is_alt_race, delta` |
| `telemetry/race_NN.parquet` | ~600/race | `x, y, speed, throttle, brake, gear, driver_code` |

### Dashboard (`dashboard/app.py`)

Single-page Streamlit app with a race slider (1–24) controlling all panels. Layout:
- **Top**: title + slider
- **Left**: track map (telemetry X/Y scatter) + fastest lap trace + lap time
- **Center**: championship progression line chart (NOR/VER/PIA) + championship probability pie chart
- **Right**: race summary — top 3 finishers, weather stats, narrative insight
- **Bottom right**: "Alternate Reality" button (active from Round 14 onward) — swaps center charts to `alternate_reality.csv` data

Narrative phases (`narrative_phase` column) drive the story framing: Season Start (1–4), McLaren Dominance (5–9), Three-Way Battle (10–13), Turning Point (14), Finale (15–24).

### Environment

- Python 3.13.5 (Anaconda). `requirements.txt` uses `>=` constraints — do **not** pin `numpy==1.26.4` (no Python 3.13 wheel).
- Installed: FastF1 3.8.2, pandas 2.2.3, numpy 2.1.3, plotly **6.5.2** (not 5.x), streamlit 1.45.1, pyarrow 19.0.0.
- Plotly is 6.x — if graph_objects patterns from 5.x docs fail, check the 6.x changelog.
- `data/raw/` and `cache/` are gitignored. `data/processed/` is the source of truth for the dashboard.
