from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ── Constants ─────────────────────────────────────────────────────────────────
CONTENDERS = ["NOR", "VER", "PIA"]

DRIVER_COLORS = {
    "NOR": "#FF8000",
    "VER": "#3671C6",
    "PIA": "#E8002D",
}

DRIVER_FILL_COLORS = {
    "NOR": "rgba(255,128,0,0.08)",
    "VER": "rgba(54,113,198,0.08)",
    "PIA": "rgba(232,0,45,0.07)",
}

DRIVER_FULLNAME = {
    "NOR": "Lando Norris",
    "VER": "Max Verstappen",
    "PIA": "Oscar Piastri",
}

TEAM_COLORS = {
    "McLaren":          "#FF8000",
    "Red Bull Racing":  "#3671C6",
    "Mercedes":         "#27F4D2",
    "Ferrari":          "#E8002D",
    "Williams":         "#64C4FF",
    "Aston Martin":     "#229971",
    "Alpine":           "#FF87BC",
    "Haas F1 Team":     "#B6BABD",
    "Kick Sauber":      "#52E252",
    "Racing Bulls":     "#6692FF",
}

PHASE_COLORS = {
    "Season Start":      "#4A90D9",
    "McLaren Dominance": "#FF8000",
    "Three-Way Battle":  "#9B59B6",
    "Turning Point":     "#E74C3C",
    "Finale":            "#2ECC71",
}

PHASE_BANDS = [
    (0.5,  4.5,  (74,  144, 217), "START"),
    (4.5,  9.5,  (255, 128, 0),   "McLAREN"),
    (9.5,  13.5, (155, 89,  182), "BATTLE"),
    (13.5, 14.5, (231, 76,  60),  "PIVOT"),
    (14.5, 24.5, (46,  204, 113), "FINALE"),
]

PHASE_TIMELINE_SEGS = [
    (1,  4,  (74,  144, 217), "START"),
    (5,  9,  (255, 128, 0),   "McLAREN"),
    (10, 13, (155, 89,  182), "BATTLE"),
    (14, 14, (231, 76,  60),  "PIVOT"),
    (15, 24, (46,  204, 113), "FINALE"),
]

INCIDENT_ROUNDS = [
    (14, "Budapest"),
    (20, "Mexico"),
    (21, "São Paulo"),
    (23, "Qatar"),
]

MEDALS = ["🥇", "🥈", "🥉"]

SCENARIOS_DISPLAY = [
    {
        "id": "hungary_ver", "name": "The Budapest Reversal",
        "hook": "VER +30 pt swing", "round": 14, "short": "Budapest",
        "what_happened": "Norris wins Hungary; Verstappen finishes P9 — just 2 championship points.",
        "what_changed":  "Verstappen wins, Norris takes P2 — a 30-point swing in a single afternoon.",
    },
    {
        "id": "mexico_ver", "name": "Mexico City Redemption",
        "hook": "VER +17 pt swing", "round": 20, "short": "Mexico",
        "what_happened": "Norris wins Mexico City, extending his lead to 43 points with 4 races left.",
        "what_changed":  "Verstappen wins, cutting the gap to 26 — still alive into the final stretch.",
    },
    {
        "id": "saopaulo_ver", "name": "Interlagos What-If",
        "hook": "VER +17 pt swing", "round": 21, "short": "São Paulo",
        "what_happened": "Verstappen starts P19 after grid penalties and recovers to only 15 points.",
        "what_changed":  "Verstappen wins from the back — the gap drops to 9 points before Las Vegas.",
    },
    {
        "id": "qatar_kimi", "name": "The Kimi Moment",
        "hook": "2 pts · margin becomes 3", "round": 23, "short": "Qatar",
        "what_happened": "Kimi Antonelli yields P4 to Norris in Qatar — a 2-point gift that sealed the title.",
        "what_changed":  "Antonelli holds position; Norris stays P5 — final margin becomes 3 pts, not 5.",
    },
]


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_race_cards() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "race_cards.csv")


@st.cache_data
def load_contender_progression() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "contender_progression.csv")


@st.cache_data
def load_predictor_probs() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "predictor_probs.csv")


@st.cache_data
def load_alternate_scenarios() -> pd.DataFrame:
    path = PROCESSED_DIR / "alternate_scenarios.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_telemetry(round_num: int) -> pd.DataFrame:
    path = PROCESSED_DIR / "telemetry" / f"race_{round_num:02d}.parquet"
    return pd.read_parquet(path)


@st.cache_data
def load_driver_images() -> dict:
    path = PROCESSED_DIR / "driver_images.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path).dropna(subset=["headshot_url"])
    return dict(zip(df["driver_code"], df["headshot_url"]))


# ── Utilities ─────────────────────────────────────────────────────────────────
def format_lap_time_ms(ms: int) -> str:
    minutes = ms // 60000
    seconds = (ms % 60000) / 1000.0
    return f"{minutes}:{seconds:06.3f}"


def format_lap_time_str(timedelta_str: str) -> str:
    td = pd.to_timedelta(timedelta_str)
    return format_lap_time_ms(int(td.total_seconds() * 1000))


def get_team_color(team_name: str) -> str:
    return TEAM_COLORS.get(team_name, "#555555")


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── CSS ───────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

    /* ── Global ──────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, sans-serif;
    }
    .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }

    /* ── F1 red identity stripe at very top ──────────── */
    .stApp::before {
        content: '';
        display: block;
        height: 3px;
        background: linear-gradient(90deg, #E10600 0%, #FF8000 55%, #3671C6 100%);
        position: fixed;
        top: 0; left: 0; right: 0;
        z-index: 9999;
    }

    /* ── Slider label & track ─────────────────────────── */
    .slider-label {
        font-size: 0.58rem;
        font-weight: 800;
        color: #E10600;
        letter-spacing: 0.20em;
        text-transform: uppercase;
        margin-bottom: -10px;
        display: block;
    }
    [data-testid="stSlider"] { padding: 0 2px; }
    [data-baseweb="slider"] [role="slider"] {
        background-color: #E10600 !important;
        border-color: #E10600 !important;
        box-shadow: 0 0 0 3px rgba(225,6,0,0.20) !important;
    }

    /* ── Section labels ──────────────────────────────── */
    .section-label {
        font-size: 0.58rem;
        font-weight: 800;
        color: #E10600;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        display: block;
        margin-bottom: 3px;
        margin-top: 6px;
    }

    /* ── Page header ─────────────────────────────────── */
    .page-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 6px 0 10px;
        border-bottom: 1px solid rgba(255,255,255,0.07);
        margin-bottom: 8px;
    }
    .header-f1 {
        font-size: 1.30rem;
        font-weight: 900;
        letter-spacing: -0.02em;
        color: #E10600;
        line-height: 1;
    }
    .header-title {
        font-size: 1.05rem;
        font-weight: 900;
        letter-spacing: -0.01em;
        color: #ffffff;
        line-height: 1;
    }
    .header-round {
        font-size: 0.68rem;
        color: #444;
        margin-left: auto;
        letter-spacing: 0.06em;
    }

    /* ── Phase badge ─────────────────────────────────── */
    .phase-banner {
        padding: 3px 10px;
        border-radius: 3px;
        font-weight: 800;
        font-size: 0.60rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        display: inline-block;
        color: white;
    }

    /* ── Season timeline strip ───────────────────────── */
    .phase-timeline {
        display: flex;
        height: 20px;
        border-radius: 3px;
        overflow: hidden;
        margin: 0 0 8px;
        gap: 1px;
    }
    .phase-seg {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.50rem;
        font-weight: 800;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        overflow: hidden;
        white-space: nowrap;
    }

    /* ── Race stats (left panel) ─────────────────────── */
    .race-report-header {
        margin: 8px 0 6px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255,255,255,0.07);
    }
    .race-report-eyebrow {
        font-size: 0.58rem;
        color: #E10600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-weight: 800;
        margin-bottom: 3px;
    }
    .race-report-title {
        font-size: 1.06rem;
        font-weight: 900;
        line-height: 1.15;
        margin: 2px 0 4px;
        letter-spacing: -0.01em;
        color: #ffffff;
    }
    .race-report-dateline {
        font-size: 0.68rem;
        color: #666;
    }

    /* ── Driver headshots ────────────────────────────── */
    .podium-photo-row {
        display: flex;
        gap: 6px;
        margin: 8px 0 10px;
        justify-content: flex-start;
    }
    .podium-photo-cell {
        text-align: center;
        min-width: 58px;
    }
    .driver-headshot {
        width: 52px;
        height: 52px;
        border-radius: 50%;
        border: 2px solid #555;
        object-fit: cover;
        display: block;
        margin: 0 auto;
    }
    .driver-headshot-placeholder {
        width: 52px;
        height: 52px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.80rem;
        font-weight: 900;
        color: white;
        margin: 0 auto;
    }
    .driver-code-below {
        font-size: 0.62rem;
        font-weight: 800;
        color: #bbb;
        margin-top: 3px;
        letter-spacing: 0.05em;
    }
    .driver-pos-below {
        font-size: 0.54rem;
        color: #555;
        margin-top: 1px;
    }

    /* ── Podium cards ─────────────────────────────────── */
    .podium-card {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 5px 8px;
        background: rgba(255,255,255,0.025);
        border-radius: 4px;
        margin-bottom: 4px;
    }
    .podium-medal { font-size: 0.90rem; min-width: 18px; }
    .podium-code  { font-weight: 900; font-size: 0.88rem; min-width: 36px; letter-spacing: 0.03em; }
    .podium-name  { font-size: 0.68rem; color: #999; flex: 1; }
    .podium-team  { font-size: 0.60rem; text-align: right; max-width: 82px; }

    /* ── Stat pills ───────────────────────────────────── */
    .stat-pills {
        display: flex;
        gap: 4px;
        flex-wrap: wrap;
        margin: 6px 0 4px;
    }
    .stat-pill {
        background: rgba(255,255,255,0.05);
        border-radius: 20px;
        padding: 3px 8px;
        font-size: 0.64rem;
        color: #aaa;
    }

    /* ── Narrative block ──────────────────────────────── */
    .narrative-block {
        border-left: 2px solid rgba(255,255,255,0.08);
        padding: 5px 9px;
        font-size: 0.73rem;
        color: #aaa;
        line-height: 1.60;
        margin: 6px 0 2px;
    }

    /* ── Max section header (right panel) ────────────── */
    .max-section-header {
        border-left: 3px solid #3671C6;
        padding-left: 10px;
        margin-bottom: 10px;
        margin-top: 2px;
    }
    .max-section-eyebrow {
        font-size: 0.56rem;
        color: #E10600;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        font-weight: 800;
        display: block;
        margin-bottom: 2px;
    }
    .max-section-title {
        font-size: 0.98rem;
        font-weight: 900;
        color: #ffffff;
        display: block;
        line-height: 1.15;
    }

    /* ── Scenario cards ───────────────────────────────── */
    .scenario-card {
        border-radius: 5px;
        margin-bottom: 3px;
        transition: background 0.15s;
    }
    .scenario-card-active {
        box-shadow: 0 0 0 1px #3671C6, 0 0 16px rgba(54,113,198,0.22);
    }
    .scenario-locked {
        opacity: 0.28;
        pointer-events: none;
    }

    /* ── Metrics ──────────────────────────────────────── */
    [data-testid="stMetricValue"] {
        font-size: 1.20rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.68rem !important;
        color: #666 !important;
    }

    /* ── Scrollable right panel ───────────────────────── */
    [data-testid="column"]:last-child {
        overflow-y: auto;
        max-height: 92vh;
    }

    /* ── Divider ──────────────────────────────────────── */
    .panel-divider {
        height: 1px;
        background: rgba(255,255,255,0.06);
        margin: 8px 0;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Charts ────────────────────────────────────────────────────────────────────
def build_track_map(telem_df: pd.DataFrame) -> go.Figure:
    x     = telem_df["x"].values
    y     = telem_df["y"].values
    speed = telem_df["speed"].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(
            color=speed,
            colorscale="RdYlGn",
            cmin=float(speed.min()),
            cmax=float(speed.max()),
            size=3,
            showscale=False,
        ),
        hoverinfo="none",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[float(x[0])], y=[float(y[0])],
        mode="markers",
        marker=dict(symbol="star", size=13, color="white",
                    line=dict(color="#222", width=1)),
        showlegend=False,
        hovertemplate="Start / Finish<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#0a0a14",
        plot_bgcolor="#0a0a14",
        margin=dict(l=2, r=2, t=2, b=2),
        height=200,
        xaxis=dict(scaleanchor="y", scaleratio=1,
                   showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )
    return fig


def build_championship_chart(
    cp_df: pd.DataFrame,
    selected_round: int,
    alt_scen_df: pd.DataFrame,
    selected_scenario: "str | None",
) -> go.Figure:
    fig = go.Figure()

    for x0, x1, (r, g, b), label in PHASE_BANDS:
        mid = (x0 + x1) / 2
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor=f"rgba({r},{g},{b},0.07)",
            layer="below", line_width=0,
        )
        fig.add_annotation(
            x=mid, y=0.995, yref="paper",
            text=label, showarrow=False,
            font=dict(size=7, color=f"rgba({r},{g},{b},0.40)"),
            yanchor="top", xanchor="center",
        )

    filtered = cp_df[cp_df["round_number"] <= selected_round]
    for driver in CONTENDERS:
        d = filtered[filtered["driver_code"] == driver].sort_values("round_number")
        color      = DRIVER_COLORS[driver]
        fill_color = DRIVER_FILL_COLORS[driver]
        fig.add_trace(go.Scatter(
            x=d["round_number"].tolist(),
            y=d["cumulative_points"].tolist(),
            mode="lines+markers",
            name=DRIVER_FULLNAME[driver],
            line=dict(color=color, width=2.5),
            marker=dict(size=4, color=color),
            fill="tozeroy",
            fillcolor=fill_color,
            hovertemplate=(
                f"<b>{DRIVER_FULLNAME[driver]}</b><br>"
                "Round %{x} — %{customdata[0]}<br>"
                "Total: <b>%{y} pts</b><br>"
                "This race: %{customdata[1]} pts  (P%{customdata[2]})<extra></extra>"
            ),
            customdata=d[["event_name", "race_points", "position"]].values,
        ))

    if not selected_scenario:
        for rnd, _ in INCIDENT_ROUNDS:
            if rnd <= selected_round:
                fig.add_vline(
                    x=rnd, line_dash="dot",
                    line_color="rgba(255,200,0,0.18)", line_width=1,
                )

    if selected_scenario and not alt_scen_df.empty:
        scen_data = alt_scen_df[alt_scen_df["scenario_id"] == selected_scenario]
        if not scen_data.empty:
            turning_pt = int(scen_data["turning_point"].iloc[0])
            alt_f = scen_data[scen_data["round_number"] <= selected_round]
            for driver in CONTENDERS:
                da = alt_f[alt_f["driver_code"] == driver].sort_values("round_number")
                if da.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=da["round_number"].tolist(),
                    y=da["alt_cumulative_points"].tolist(),
                    mode="lines",
                    name=f"{driver} (Alt)",
                    line=dict(color=DRIVER_COLORS[driver], width=1.5, dash="dash"),
                    opacity=0.65,
                    hovertemplate=(
                        f"<b>{driver} Alternate</b><br>"
                        "Round %{x}<br>"
                        "Alt pts: <b>%{y}</b><extra></extra>"
                    ),
                ))
            fig.add_vline(
                x=turning_pt, line_dash="dot",
                line_color="rgba(255,210,0,0.60)", line_width=1.5,
            )

    fig.add_vline(
        x=selected_round, line_dash="dot",
        line_color="rgba(255,255,255,0.35)", line_width=1.5,
    )

    fig.update_layout(
        paper_bgcolor="#0d0d1f",
        plot_bgcolor="#0d0d1f",
        height=285,
        margin=dict(l=10, r=10, t=22, b=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0,
            font=dict(color="white", size=9),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            title=dict(text="Round", font=dict(color="#444", size=10)),
            showgrid=True, gridcolor="rgba(255,255,255,0.04)",
            color="#555", tickmode="linear", dtick=3,
            range=[0.5, 24.5],
        ),
        yaxis=dict(
            title=dict(text="Points", font=dict(color="#444", size=10)),
            showgrid=True, gridcolor="rgba(255,255,255,0.04)",
            color="#555",
        ),
        font=dict(color="white"),
    )
    return fig


def build_probability_bars(pp_df: pd.DataFrame, selected_round: int) -> go.Figure:
    rnd = pp_df[pp_df["round_number"] == selected_round]
    drivers, probs, colors = [], [], []
    for driver in reversed(CONTENDERS):
        row = rnd[rnd["driver_code"] == driver]
        if row.empty:
            continue
        prob = float(row["championship_probability"].iloc[0]) * 100
        drivers.append(driver)
        probs.append(max(prob, 0.0))
        colors.append(DRIVER_COLORS[driver])

    if not probs or sum(probs) == 0:
        probs = [100 / len(CONTENDERS)] * len(CONTENDERS)

    fig = go.Figure(go.Bar(
        x=probs,
        y=drivers,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{p:.1f}%" for p in probs],
        textposition="outside",
        textfont=dict(color="white", size=11, family="monospace"),
        hovertemplate="%{y}: <b>%{x:.1f}%</b><extra></extra>",
        cliponaxis=False,
    ))
    fig.update_layout(
        paper_bgcolor="#0d0d1f",
        plot_bgcolor="#0d0d1f",
        height=130,
        margin=dict(l=10, r=70, t=6, b=6),
        xaxis=dict(range=[0, 120],
                   showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, color="white",
                   tickfont=dict(size=12, family="monospace"), automargin=True),
        font=dict(color="white"),
        bargap=0.40,
    )
    return fig


def build_scenario_comparison_chart(
    pp_df: pd.DataFrame,
    alt_scen_df: pd.DataFrame,
    scenario_id: str,
    selected_round: int,
) -> go.Figure:
    scen_data = alt_scen_df[alt_scen_df["scenario_id"] == scenario_id]
    if scen_data.empty:
        return build_probability_bars(pp_df, selected_round)

    turning_pt    = int(scen_data["turning_point"].iloc[0])
    scen_info     = next((s for s in SCENARIOS_DISPLAY if s["id"] == scenario_id), None)
    incident_label = scen_info["short"] if scen_info else "Incident"

    fig = go.Figure()
    endpoints: list[tuple] = []

    for driver in CONTENDERS:
        drv_pp = pp_df[pp_df["driver_code"] == driver].sort_values("round_number")
        rows   = drv_pp[drv_pp["round_number"] <= selected_round]
        if rows.empty:
            continue
        last_x = int(rows["round_number"].iloc[-1])
        last_y = float(rows["championship_probability"].iloc[-1]) * 100
        endpoints.append((last_x, last_y, driver, DRIVER_COLORS[driver], False))
        fig.add_trace(go.Scatter(
            x=rows["round_number"].tolist(),
            y=(rows["championship_probability"] * 100).tolist(),
            mode="lines",
            name=driver,
            line=dict(color=DRIVER_COLORS[driver], width=2.5, dash="solid"),
            hovertemplate=f"<b>{driver} actual</b><br>R%{{x}} → %{{y:.1f}}%<extra></extra>",
        ))

    for driver in ["VER", "NOR"]:
        drv_alt = scen_data[scen_data["driver_code"] == driver].sort_values("round_number")
        rows    = drv_alt[drv_alt["round_number"] <= selected_round]
        if rows.empty:
            continue
        last_x = int(rows["round_number"].iloc[-1])
        last_y = float(rows["alt_championship_prob"].iloc[-1]) * 100
        endpoints.append((last_x, last_y, f"~{driver}", DRIVER_COLORS[driver], True))
        fig.add_trace(go.Scatter(
            x=rows["round_number"].tolist(),
            y=(rows["alt_championship_prob"] * 100).tolist(),
            mode="lines",
            name=f"{driver} alt",
            line=dict(color=DRIVER_COLORS[driver], width=2, dash="dash"),
            opacity=0.70,
            hovertemplate=f"<b>{driver} alt</b><br>R%{{x}} → %{{y:.1f}}%<extra></extra>",
        ))

    for lx, ly, label, color, is_alt in endpoints:
        fig.add_annotation(
            x=lx, y=ly, text=label,
            showarrow=False, xanchor="left", xshift=6,
            font=dict(color=hex_to_rgba(color, 0.60 if is_alt else 1.0),
                      size=8, family="monospace"),
            yanchor="middle",
        )

    fig.add_vline(x=turning_pt, line_width=1, line_dash="dot",
                  line_color="rgba(255,210,0,0.55)")
    fig.add_annotation(
        x=turning_pt, y=96,
        text=incident_label, showarrow=False,
        font=dict(color="rgba(255,210,0,0.65)", size=8),
        yanchor="top", xanchor="center",
    )

    fig.update_layout(
        paper_bgcolor="#0d0d1f",
        plot_bgcolor="#0d0d1f",
        height=300,
        showlegend=False,
        margin=dict(l=10, r=55, t=18, b=10),
        xaxis=dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.04)",
            color="#555", tickmode="linear", dtick=3,
            range=[0.5, 24.5],
        ),
        yaxis=dict(
            title=dict(text="Win %", font=dict(color="#444", size=10)),
            range=[0, 107],
            showgrid=True, gridcolor="rgba(255,255,255,0.04)",
            color="#555",
        ),
        font=dict(color="white"),
    )
    return fig


# ── Narrative ─────────────────────────────────────────────────────────────────
def generate_narrative(card: pd.Series, cp_df: pd.DataFrame, round_num: int) -> str:
    winner   = card["p1_driver"]
    circuit  = card["circuit_short_name"]
    rainfall = bool(card["rainfall"])
    weather  = "wet conditions" if rainfall else "dry conditions"
    phase    = card["narrative_phase"]

    rnd_cp = cp_df[cp_df["round_number"] == round_num].set_index("driver_code")

    def total(d):
        return int(rnd_cp.loc[d, "cumulative_points"]) if d in rnd_cp.index else 0

    def race_pts(d):
        return int(rnd_cp.loc[d, "race_points"]) if d in rnd_cp.index else 0

    standings = sorted(
        [(d, total(d)) for d in CONTENDERS],
        key=lambda x: x[1], reverse=True,
    )
    leader, leader_pts = standings[0]
    second, second_pts = standings[1]
    gap = leader_pts - second_pts

    winner_name = DRIVER_FULLNAME.get(winner, winner)

    if winner in CONTENDERS:
        s1 = (
            f"{winner_name} takes victory in {weather} at {circuit}, "
            f"scoring {race_pts(winner)} points."
        )
    else:
        parts = [f"{d} +{race_pts(d)}" for d in CONTENDERS if race_pts(d) > 0]
        pts_str = ", ".join(parts) if parts else "the title contenders scoring nothing"
        s1 = f"{winner_name} wins at {circuit} in {weather} — {pts_str}."

    s2 = (
        f"{DRIVER_FULLNAME[leader]} leads with {leader_pts} pts, "
        f"{gap} ahead of {DRIVER_FULLNAME[second]}."
    )

    remaining = 24 - round_num
    if remaining == 0:
        s3 = f"The 2025 season is over — {DRIVER_FULLNAME[leader]} is World Champion!"
    elif phase == "Turning Point":
        s3 = "This race reshapes the title fight — everything changes from here."
    elif remaining <= 3:
        s3 = f"Only {remaining} race{'s' if remaining > 1 else ''} remain — the pressure is immense."
    elif gap <= 10:
        s3 = f"Separated by just {gap} pts — every single point matters."
    else:
        s3 = f"{remaining} races still to go — the championship fight is far from over."

    return f"{s1} {s2} {s3}"


# ── Phase timeline strip ──────────────────────────────────────────────────────
def render_phase_timeline(selected_round: int):
    segs_html = ""
    for start, end, (r, g, b), label in PHASE_TIMELINE_SEGS:
        active    = start <= selected_round <= end
        bg_alpha  = 0.32 if active else 0.07
        bdr_alpha = 0.90 if active else 0.25
        txt_alpha = 0.95 if active else 0.30
        width     = end - start + 1
        segs_html += (
            f'<div class="phase-seg" style="flex:{width};'
            f'background:rgba({r},{g},{b},{bg_alpha});'
            f'border-top:2px solid rgba({r},{g},{b},{bdr_alpha});'
            f'color:rgba({r},{g},{b},{txt_alpha});">'
            f'{label if width > 1 else "·"}'
            f'</div>'
        )
    st.markdown(
        f'<div class="phase-timeline">{segs_html}</div>',
        unsafe_allow_html=True,
    )


# ── Left panel — Track map + Race stats ──────────────────────────────────────
def render_left_panel(
    telem_df: pd.DataFrame,
    card: pd.Series,
    cp_df: pd.DataFrame,
    driver_images: dict,
    selected_round: int,
):
    # Track map
    st.markdown('<span class="section-label">Track Map · Speed</span>',
                unsafe_allow_html=True)
    st.plotly_chart(build_track_map(telem_df), use_container_width=True)

    # Race header
    phase_color = PHASE_COLORS.get(card["narrative_phase"], "#666")
    st.markdown(
        f'<div class="race-report-header">'
        f'<div class="race-report-eyebrow">'
        f'Round {selected_round} &nbsp;·&nbsp; {card["narrative_phase"]}</div>'
        f'<div class="race-report-title">{card["event_name"]}</div>'
        f'<div class="race-report-dateline">'
        f'{card["country"]} &nbsp;·&nbsp; {card["session_date"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Driver headshot row
    podium_drivers = [card["p1_driver"], card["p2_driver"], card["p3_driver"]]
    podium_teams   = [card["p1_team"],   card["p2_team"],   card["p3_team"]]
    photo_cells = ""
    for pos_label, driver_code, team_name in zip(["P1", "P2", "P3"], podium_drivers, podium_teams):
        if not driver_code or pd.isna(driver_code):
            continue
        team_color = get_team_color(team_name)
        url = driver_images.get(driver_code)
        if url:
            img_html = (
                f'<img src="{url}" class="driver-headshot" '
                f'style="border-color:{team_color};" />'
            )
        else:
            initials = driver_code[:2]
            img_html = (
                f'<div class="driver-headshot-placeholder" '
                f'style="background:{team_color}22;border:2px solid {team_color};">'
                f'{initials}</div>'
            )
        photo_cells += (
            f'<div class="podium-photo-cell">'
            f'{img_html}'
            f'<div class="driver-code-below" style="color:{team_color};">{driver_code}</div>'
            f'<div class="driver-pos-below">{pos_label}</div>'
            f'</div>'
        )
    st.markdown(
        f'<div class="podium-photo-row">{photo_cells}</div>',
        unsafe_allow_html=True,
    )

    # Podium cards
    st.markdown('<span class="section-label">Race Result</span>', unsafe_allow_html=True)
    podium_html = ""
    for medal, driver_code, team_name in zip(MEDALS, podium_drivers, podium_teams):
        if not driver_code or pd.isna(driver_code):
            continue
        full_name  = DRIVER_FULLNAME.get(driver_code, driver_code)
        team_color = get_team_color(team_name)
        podium_html += (
            f'<div class="podium-card" style="border-left:3px solid {team_color};">'
            f'<span class="podium-medal">{medal}</span>'
            f'<span class="podium-code">{driver_code}</span>'
            f'<span class="podium-name">{full_name}</span>'
            f'<span class="podium-team" style="color:{team_color};">{team_name}</span>'
            f'</div>'
        )
    st.markdown(podium_html, unsafe_allow_html=True)

    # Conditions stat pills
    weather_icon = "🌧" if bool(card["rainfall"]) else "☀️"
    pills_html = '<div class="stat-pills">'
    pills_html += f'<span class="stat-pill">{weather_icon} {card["weather_description"]}</span>'
    track_t = card.get("track_temp_avg")
    air_t   = card.get("air_temp_avg")
    if pd.notna(track_t):
        pills_html += f'<span class="stat-pill">🌡 {float(track_t):.0f}°C track</span>'
    if pd.notna(air_t):
        pills_html += f'<span class="stat-pill">💨 {float(air_t):.0f}°C air</span>'
    if pd.notna(card.get("pole_driver")):
        pills_html += f'<span class="stat-pill">🏁 Pole: {card["pole_driver"]}</span>'
    if pd.notna(card.get("fastest_lap_driver")) and pd.notna(card.get("fastest_lap_time")):
        try:
            fl_str = format_lap_time_str(str(card["fastest_lap_time"]))
            pills_html += f'<span class="stat-pill">⚡ {card["fastest_lap_driver"]} {fl_str}</span>'
        except Exception:
            pass
    pills_html += '</div>'
    st.markdown(pills_html, unsafe_allow_html=True)

    # Narrative
    narrative = generate_narrative(card, cp_df, selected_round)
    st.markdown(
        f'<div class="narrative-block" style="border-left-color:{phase_color}55;">'
        f'{narrative}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Center panel — Championship charts ───────────────────────────────────────
def render_center_panel(
    cp_df: pd.DataFrame,
    pp_df: pd.DataFrame,
    alt_scen_df: pd.DataFrame,
    selected_round: int,
    selected_scenario: "str | None",
):
    st.markdown('<span class="section-label">Championship Progression</span>',
                unsafe_allow_html=True)
    st.plotly_chart(
        build_championship_chart(cp_df, selected_round, alt_scen_df, selected_scenario),
        use_container_width=True,
    )

    st.markdown('<span class="section-label">Title Win Probability</span>',
                unsafe_allow_html=True)
    if selected_scenario and not alt_scen_df.empty:
        st.plotly_chart(
            build_scenario_comparison_chart(pp_df, alt_scen_df, selected_scenario, selected_round),
            use_container_width=True,
        )
    else:
        st.plotly_chart(
            build_probability_bars(pp_df, selected_round),
            use_container_width=True,
        )


# ── Right panel — Max's Road to the Title ────────────────────────────────────
def render_right_panel(
    pp_df: pd.DataFrame,
    alt_scen_df: pd.DataFrame,
    selected_round: int,
    selected_scenario: "str | None",
):
    # Section header
    st.markdown(
        '<div class="max-section-header">'
        '<span class="max-section-eyebrow">Alternate Reality</span>'
        '<span class="max-section-title">Max\'s Road to the Title</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # 4 scenario cards in 2×2 grid
    c1, c2 = st.columns(2)
    col_map = [c1, c2, c1, c2]
    for i, scen in enumerate(SCENARIOS_DISPLAY):
        is_active = selected_scenario == scen["id"]
        is_locked = selected_round < scen["round"]

        # p_incident badge
        p_inc_html = ""
        if not alt_scen_df.empty:
            s_rows = alt_scen_df[alt_scen_df["scenario_id"] == scen["id"]]
            if not s_rows.empty and "p_incident" in s_rows.columns:
                p_i = float(s_rows["p_incident"].iloc[0]) * 100
                p_inc_html = (
                    f'<div style="color:#555;font-size:0.60rem;margin-top:2px;">'
                    f'P(incident): {p_i:.1f}%</div>'
                )

        border = "2px solid #3671C6" if is_active else "1px solid #222235"
        bg     = "rgba(54,113,198,0.08)" if is_active else "rgba(255,255,255,0.015)"
        glow   = ' scenario-card-active' if is_active else ''
        locked = ' scenario-locked' if is_locked else ''
        lock_note = (
            f'<div style="color:#444;font-size:0.58rem;margin-top:2px;">'
            f'Locked until R{scen["round"]}</div>'
        ) if is_locked else ""

        with col_map[i]:
            st.markdown(
                f'<div class="scenario-card{glow}{locked}" style="border:{border};'
                f'background:{bg};padding:7px 9px;border-radius:5px;margin-bottom:2px;">'
                f'<div style="font-weight:800;font-size:0.76rem;">{scen["name"]}</div>'
                f'<div style="color:#555;font-size:0.63rem;margin-top:1px;">'
                f'R{scen["round"]} · {scen["hook"]}</div>'
                f'{p_inc_html}{lock_note}'
                f'</div>',
                unsafe_allow_html=True,
            )
            if not is_locked:
                if st.button(
                    "Deselect" if is_active else "Explore",
                    key=f"scen_{scen['id']}",
                    use_container_width=True,
                ):
                    st.session_state.selected_scenario = None if is_active else scen["id"]
                    st.rerun()

    # "Everything Goes Right" combined card — full width
    all_active = selected_scenario == "all_ver"
    all_border = "2px solid #FF8000" if all_active else "1px solid #2a2020"
    all_bg     = "rgba(255,128,0,0.07)" if all_active else "rgba(255,255,255,0.015)"
    all_glow   = ' scenario-card-active' if all_active else ''
    compound_badge = ""
    if not alt_scen_df.empty:
        av_rows = alt_scen_df[alt_scen_df["scenario_id"] == "all_ver"]
        if not av_rows.empty and "p_incident" in av_rows.columns:
            p_c = float(av_rows["p_incident"].iloc[0])
            if p_c > 0:
                one_in = round(1.0 / p_c)
                compound_badge = (
                    f'<span style="color:#FF8000;font-size:0.68rem;font-weight:700;">'
                    f' · {p_c*100:.2f}% · 1 in {one_in}</span>'
                )
    is_all_locked = selected_round < 14
    all_locked_cls = ' scenario-locked' if is_all_locked else ''
    st.markdown(
        f'<div class="scenario-card{all_glow}{all_locked_cls}" style="border:{all_border};'
        f'background:{all_bg};padding:7px 9px;border-radius:5px;margin-bottom:2px;">'
        f'<div style="font-weight:800;font-size:0.76rem;">'
        f'Everything Goes Right{compound_badge}</div>'
        f'<div style="color:#555;font-size:0.63rem;margin-top:1px;">'
        f'All 4 incidents · VER becomes champion</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if not is_all_locked:
        if st.button(
            "Deselect" if all_active else "Explore",
            key="scen_all_ver",
            use_container_width=True,
        ):
            st.session_state.selected_scenario = None if all_active else "all_ver"
            st.rerun()

    # ── Active scenario detail ─────────────────────────
    if selected_scenario and not alt_scen_df.empty:
        scen_data = alt_scen_df[alt_scen_df["scenario_id"] == selected_scenario]
        if not scen_data.empty:
            st.markdown('<div class="panel-divider"></div>', unsafe_allow_html=True)

            scen_name = scen_data["scenario_name"].iloc[0]
            scen_info = next((s for s in SCENARIOS_DISPLAY if s["id"] == selected_scenario), None)

            # What happened / What if
            st.markdown('<span class="section-label">The Story</span>', unsafe_allow_html=True)
            if scen_info:
                st.markdown(
                    f'<div style="background:#0e0e20;padding:10px 12px;border-radius:5px;'
                    f'border-left:3px solid #222244;margin-bottom:6px;">'
                    f'<div style="color:#E10600;font-size:0.56rem;text-transform:uppercase;'
                    f'letter-spacing:0.12em;font-weight:800;margin-bottom:3px;">What happened</div>'
                    f'<div style="font-size:0.74rem;color:#888;margin-bottom:8px;'
                    f'line-height:1.50;">{scen_info["what_happened"]}</div>'
                    f'<div style="color:#4a90d9;font-size:0.56rem;text-transform:uppercase;'
                    f'letter-spacing:0.12em;font-weight:800;margin-bottom:3px;">What if</div>'
                    f'<div style="font-size:0.82rem;color:#e8f0ff;font-weight:700;'
                    f'line-height:1.50;">{scen_info["what_changed"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            elif selected_scenario == "all_ver":
                st.markdown(
                    f'<div style="background:#0e0e20;padding:10px 12px;border-radius:5px;'
                    f'border-left:3px solid #222244;margin-bottom:6px;">'
                    f'<div style="color:#E10600;font-size:0.56rem;text-transform:uppercase;'
                    f'letter-spacing:0.12em;font-weight:800;margin-bottom:3px;">What happened</div>'
                    f'<div style="font-size:0.74rem;color:#888;margin-bottom:8px;'
                    f'line-height:1.50;">Four separate incidents across the season '
                    f'each cost Verstappen critical points.</div>'
                    f'<div style="color:#4a90d9;font-size:0.56rem;text-transform:uppercase;'
                    f'letter-spacing:0.12em;font-weight:800;margin-bottom:3px;">What if</div>'
                    f'<div style="font-size:0.82rem;color:#e8f0ff;font-weight:700;'
                    f'line-height:1.50;">All four go Verstappen\'s way — Budapest, '
                    f'Mexico, São Paulo, and Qatar.</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Alt final standings
            final = scen_data[scen_data["round_number"] == 24]
            if not final.empty:
                ver_row = final[final["driver_code"] == "VER"]
                nor_row = final[final["driver_code"] == "NOR"]
                ver_alt = float(ver_row["alt_cumulative_points"].iloc[0]) if not ver_row.empty else 0
                nor_alt = float(nor_row["alt_cumulative_points"].iloc[0]) if not nor_row.empty else 0
                ver_wins      = ver_alt > nor_alt
                outcome       = "Verstappen becomes World Champion!" if ver_wins else "Norris still wins the title."
                margin        = abs(ver_alt - nor_alt)
                outcome_color = "#2ECC71" if ver_wins else "#FF8000"
                st.markdown(
                    f'<div style="background:#111128;padding:9px 11px;border-radius:5px;'
                    f'border-left:3px solid #3671C6;margin-bottom:6px;">'
                    f'<div style="font-weight:800;color:#6fa8ff;font-size:0.76rem;'
                    f'margin-bottom:5px;">{scen_name}</div>'
                    f'<div style="font-size:0.74rem;margin-bottom:4px;">'
                    f'VER: <b>{ver_alt:.0f} pts</b>'
                    f'&nbsp;&nbsp;·&nbsp;&nbsp;'
                    f'NOR: <b>{nor_alt:.0f} pts</b></div>'
                    f'<div style="color:{outcome_color};font-size:0.80rem;font-weight:800;'
                    f'margin-bottom:2px;">{outcome}</div>'
                    f'<div style="color:#444;font-size:0.66rem;">Margin: {margin:.0f} pts</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Compound probability block (all_ver only)
            if selected_scenario == "all_ver":
                p_compound = float(scen_data["p_incident"].iloc[0])
                if p_compound > 0:
                    one_in_n = round(1.0 / p_compound)
                    ind_rows = []
                    for s in SCENARIOS_DISPLAY:
                        s_data = alt_scen_df[alt_scen_df["scenario_id"] == s["id"]]
                        if not s_data.empty:
                            p_s = float(s_data["p_incident"].iloc[0]) * 100
                            ind_rows.append(
                                f'<div style="display:flex;justify-content:space-between;'
                                f'font-size:0.69rem;padding:2px 0;">'
                                f'<span style="color:#666;">{s["name"]}</span>'
                                f'<span style="color:#bbb;font-weight:700;">{p_s:.1f}%</span>'
                                f'</div>'
                            )
                    st.markdown(
                        f'<div style="background:#140e05;padding:9px 11px;border-radius:5px;'
                        f'border-left:3px solid #FF8000;margin-bottom:4px;">'
                        f'<div style="font-weight:800;color:#FF8000;font-size:0.74rem;'
                        f'margin-bottom:6px;">Probability all 4 go right</div>'
                        f'{"".join(ind_rows)}'
                        f'<div style="border-top:1px solid #2a2005;margin-top:6px;'
                        f'padding-top:5px;display:flex;justify-content:space-between;'
                        f'align-items:baseline;">'
                        f'<span style="color:#555;font-size:0.66rem;">Combined</span>'
                        f'<span style="color:#FF8000;font-weight:900;font-size:0.90rem;">'
                        f'{p_compound*100:.2f}%'
                        f'<span style="font-size:0.66rem;font-weight:400;color:#666;">'
                        f' · 1 in {one_in_n}</span></span>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # VER probability delta
            cur     = scen_data[scen_data["round_number"] == selected_round]
            ver_cur = cur[cur["driver_code"] == "VER"]
            if not ver_cur.empty:
                alt_prob   = float(ver_cur["alt_championship_prob"].iloc[0]) * 100
                act_prob   = float(ver_cur["actual_championship_prob"].iloc[0]) * 100
                delta_prob = alt_prob - act_prob
                sign       = "+" if delta_prob >= 0 else ""
                st.caption(
                    f"VER at R{selected_round}: "
                    f"{act_prob:.1f}% → **{alt_prob:.1f}%** ({sign}{delta_prob:.1f}%)"
                )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        layout="wide",
        page_title="F1 2025 Alternate Reality",
        page_icon="🏎",
    )
    inject_css()

    # Load data
    race_cards    = load_race_cards()
    cp_df         = load_contender_progression()
    pp_df         = load_predictor_probs()
    alt_scen_df   = load_alternate_scenarios()
    driver_images = load_driver_images()

    # Session state
    if "selected_scenario" not in st.session_state:
        st.session_state.selected_scenario = None

    # ── Race slider ────────────────────────────────────
    rc_sorted = race_cards.sort_values("round_number")
    labels = [f"R{int(r.round_number)}: {r.event_name}" for r in rc_sorted.itertuples()]
    label_to_round = {
        f"R{int(r.round_number)}: {r.event_name}": int(r.round_number)
        for r in rc_sorted.itertuples()
    }
    st.markdown('<div class="slider-label">Select Race</div>', unsafe_allow_html=True)
    chosen_label = st.select_slider(
        "Select Race",
        options=labels,
        value=labels[-1],
        label_visibility="collapsed",
    )
    selected_round = label_to_round[chosen_label]

    # Season phase timeline strip
    render_phase_timeline(selected_round)

    # Reset scenario if slider moves before its incident round
    if st.session_state.selected_scenario:
        scen_info = next(
            (s for s in SCENARIOS_DISPLAY if s["id"] == st.session_state.selected_scenario),
            {"round": 14} if st.session_state.selected_scenario == "all_ver" else None,
        )
        if scen_info and selected_round < scen_info["round"]:
            st.session_state.selected_scenario = None

    # Current race card
    card = race_cards[race_cards["round_number"] == selected_round].iloc[0]

    # Page header
    phase_color = PHASE_COLORS.get(card["narrative_phase"], "#666")
    st.markdown(
        f'<div class="page-header">'
        f'<span class="header-f1">F1 2025</span>'
        f'<span class="header-title">Alternate Reality</span>'
        f'<span class="phase-banner" style="background:{phase_color}22;color:{phase_color};'
        f'border:1px solid {phase_color}55;">{card["narrative_phase"]}</span>'
        f'<span class="header-round">Round {selected_round} of 24</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Telemetry
    telem_df = load_telemetry(selected_round)

    # Three-column layout
    col_left, col_center, col_right = st.columns([1.35, 2.0, 1.65])

    with col_left:
        render_left_panel(telem_df, card, cp_df, driver_images, selected_round)

    with col_center:
        render_center_panel(
            cp_df, pp_df, alt_scen_df,
            selected_round, st.session_state.selected_scenario,
        )

    with col_right:
        render_right_panel(
            pp_df, alt_scen_df,
            selected_round, st.session_state.selected_scenario,
        )


if __name__ == "__main__":
    main()
