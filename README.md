# F1 2025 Alternate Reality Dashboard

An interactive Streamlit dashboard that visualizes the 2025 Formula 1 World Championship and explores a counterfactual question: **how could Max Verstappen have won the title?**

Lando Norris won the championship by just 5 points (391 vs 386). This project uses race telemetry, a machine learning model, and a Monte Carlo simulation to identify the specific race incidents that — had they gone differently — would each have been sufficient to flip the result.

---

## Features

- **Championship progression chart** with season phase annotations, area fills, and direct driver labels
- **Speed-colored telemetry track map** for every circuit, rendered from 20 Hz car position data
- **Title win probability bar chart** updated at each race round using ML-derived probabilities
- **Alternate Reality panel** with 4 counterfactual scenarios that unlock as the season progresses
- **Probability trajectory chart** overlaying actual vs. alternate championship win probabilities
- **Race slider** (R1 to R24) that updates all three panels simultaneously

---

## The 4 Alternate Scenarios

| Scenario | Round | Points Swing | P(Event) |
|---|---|---|---|
| The Budapest Reversal | R14 | +30 pts for VER | 20.25% |
| Mexico City Redemption | R20 | +17 pts for VER | 2.00% |
| Interlagos What-If | R21 | +17 pts for VER | 30.48% |
| The Kimi Moment | R23 | +2 pts for VER | 98.77% |

Any one of the first three scenarios going Verstappen's way would have been enough to win the championship.

---

## Tech Stack

- **Python 3.10+**
- **Streamlit 1.45** — dashboard framework
- **Plotly 6.x** — interactive charts
- **FastF1 3.8** — F1 telemetry and session data
- **pandas** — data processing
- **scikit-learn** — GradientBoostingClassifier
- **pyarrow** — Parquet file I/O

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/pratham-mody/F1-Alternate-Reality
cd F1-Alternate-Reality
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the dashboard

The processed data files are already included in `data/processed/`. You can launch the dashboard immediately without re-fetching any data:

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

---

## Regenerating the Data (Optional)

If you want to re-fetch the raw F1 data and rerun the ML model and Monte Carlo simulation from scratch, run the two pipeline scripts in order. This will take several minutes as it fetches all 24 race sessions and runs 50,000 simulation iterations per round.

```bash
python data/fetch_season_data.py
python data/preprocess.py
```

---

## Repository Structure

```
F1-Alternate-Reality/
├── dashboard/
│   └── app.py                  # Streamlit app entry point
├── data/
│   ├── fetch_season_data.py    # Fetches raw session data from FastF1 API
│   ├── preprocess.py           # Builds processed files, runs ML + simulation
│   ├── raw/                    # Cached raw session data (auto-generated)
│   └── processed/
│       ├── season_progression.csv      # All drivers, cumulative pts per round
│       ├── contender_progression.csv   # NOR / VER / PIA only
│       ├── race_cards.csv              # Per-race summary (podium, weather, phase)
│       ├── predictor_probs.csv         # ML championship win probability per round
│       ├── alternate_scenarios.csv     # Counterfactual trajectories
│       └── telemetry/                  # 24 Parquet files (one per race)
├── requirements.txt
└── README.md
```

---

## ML Model

A `GradientBoostingClassifier` is trained on the 2024 F1 season (479 driver-race rows) to predict finish zones: **P1**, **P2-3**, **P4-10**, or **P11+**.

**Features:** constructor points, driver points, grid position, home race flag, weather, recent form.

Predicted finish probabilities are fed into a **Gumbel-max Monte Carlo simulation** (50,000 iterations per round) to compute championship win probabilities for each driver at every stage of the season. For each counterfactual scenario, the simulation is re-run from the incident round forward with the alternate result substituted in.

---

## Key Results

- Final margin: **5 points** (NOR 391, VER 386, PIA 352)
- Hungary R14 produced a **30-point swing** — the single largest decisive moment
- Verstappen's peak actual championship win probability: **~15%**
- In the Budapest alternate reality, his probability peaks above **60%** post-R14
- The Kimi Moment: **98.77% probability** Antonelli holds position — the most avoidable incident of the season

---

## Visualization Design

Every chart decision is intentional:

- **Horizontal bars** instead of pie charts for probability comparisons (length judgment is more accurate than angle)
- **Area fills** under progression lines make the gap between drivers legible without arithmetic
- **Direct end-of-line labels** instead of a legend box — the eye never leaves the data
- **Phase band annotations** drawn directly on the championship chart encode the season narrative in the visualization itself
- **Semantic color** — NOR is always orange, VER always blue, PIA always red, across every chart

