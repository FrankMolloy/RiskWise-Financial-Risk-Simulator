from __future__ import annotations

from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import json
from datetime import datetime

from simulator import simulate
from metrics import summarise, prob_reach_goal
from market_data import fetch_stooq_close_prices, estimate_annual_return_vol


# -------------------- Config --------------------

SCENARIOS = {
    "cautious":   {"r": 0.05, "vol": 0.10},
    "balanced":   {"r": 0.07, "vol": 0.15},
    "aggressive": {"r": 0.09, "vol": 0.20},
}

PRESETS = {
    "emergency": (2, 300, 3000),
    "deposit": (5, 500, 20000),
    "retirement": (30, 400, 500000),
    "custom": (10, 200, 50000),
}

SIM_QUALITY = {
    "fast": {"sims": 1000, "seed_a": 42, "seed_b": 43},
    "standard": {"sims": 2000, "seed_a": 42, "seed_b": 43},
    "accurate": {"sims": 5000, "seed_a": 42, "seed_b": 43},
}

INFLATION_ANNUAL = 0.02


# -------------------- App --------------------

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
)
app.title = "RiskWise"
server = app.server


# -------------------- Helpers --------------------

def money(x: float) -> str:
    try:
        return f"£{float(x):,.0f}"
    except Exception:
        return "—"


def pct_band(paths: np.ndarray):
    p10 = np.percentile(paths, 10, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p90 = np.percentile(paths, 90, axis=0)
    return p10, p50, p90


def apply_crash(paths: np.ndarray, crash_pct: float, seed: int = 999) -> np.ndarray:
    """Educational stress test: one crash month per path then multiply future balances by (1-crash_pct)."""
    if crash_pct <= 0:
        return paths

    out = paths.copy()
    rng = np.random.default_rng(seed)
    n_sims, n_steps = out.shape

    lo = min(12, n_steps - 2)
    hi = max(lo + 1, n_steps - 12)
    crash_months = rng.integers(low=lo, high=hi, size=n_sims)

    factor = 1.0 - crash_pct
    for i in range(n_sims):
        m = int(crash_months[i])
        out[i, m:] *= factor

    return out


def deterministic_forecast(
    years: int,
    start_balance: float,
    monthly_contribution: float,
    expected_return_annual: float,
    inflation_annual: float,
    fee_annual: float,
) -> np.ndarray:
    """
    One-path deterministic projection using constant monthly growth.
    Returns inflation-adjusted balances (real terms).
    """
    months = int(years * 12)
    r_net_annual = expected_return_annual - fee_annual
    r_m = (1.0 + r_net_annual) ** (1.0 / 12.0) - 1.0
    infl_m = (1.0 + inflation_annual) ** (1.0 / 12.0) - 1.0

    bal = start_balance
    out = np.zeros(months + 1, dtype=float)
    out[0] = bal

    for t in range(1, months + 1):
        bal = bal * (1.0 + r_m) + monthly_contribution
        real = bal / ((1.0 + infl_m) ** t)
        out[t] = real

    return out


def build_paths_figure(months, scenario_name, p10, p50, p90, det=None,
                       p10_b=None, p50_b=None, p90_b=None, scenario_b_name=None):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=months, y=p90,
        mode="lines",
        name=f"A {scenario_name} – 90th",
        line=dict(width=1),
        hovertemplate="Month %{x}<br>P90: %{y:,.0f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=months, y=p10,
        mode="lines",
        name=f"A {scenario_name} – 10th (band)",
        fill="tonexty",
        line=dict(width=1),
        hovertemplate="Month %{x}<br>P10: %{y:,.0f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=months, y=p50,
        mode="lines",
        name=f"A {scenario_name} – Median",
        line=dict(width=3),
        hovertemplate="Month %{x}<br>Median: %{y:,.0f}<extra></extra>"
    ))

    if det is not None:
        fig.add_trace(go.Scatter(
            x=months, y=det,
            mode="lines",
            name="Deterministic (A)",
            line=dict(width=2, dash="dash"),
            hovertemplate="Month %{x}<br>Deterministic: %{y:,.0f}<extra></extra>"
        ))

    if p10_b is not None and p50_b is not None and p90_b is not None and scenario_b_name:
        fig.add_trace(go.Scatter(
            x=months, y=p90_b, mode="lines",
            name=f"B {scenario_b_name} – 90th",
            line=dict(width=1, dash="dot"),
            hovertemplate="Month %{x}<br>P90(B): %{y:,.0f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=months, y=p10_b, mode="lines",
            name=f"B {scenario_b_name} – 10th (band)",
            fill="tonexty",
            line=dict(width=1, dash="dot"),
            hovertemplate="Month %{x}<br>P10(B): %{y:,.0f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=months, y=p50_b, mode="lines",
            name=f"B {scenario_b_name} – Median",
            line=dict(width=3, dash="dot"),
            hovertemplate="Month %{x}<br>Median(B): %{y:,.0f}<extra></extra>"
        ))

    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Months",
        yaxis_title="Balance (real terms)",
        height=320,
        legend_title="Lines",
    )
    return fig


def build_hist_figure(final_a, scenario_a, median_a, goal, final_b=None, scenario_b=None, median_b=None):
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=final_a,
        nbinsx=60,
        name=f"A: {scenario_a}",
        opacity=0.75,
        marker=dict(line=dict(width=1))
    ))

    if final_b is not None:
        fig.add_trace(go.Histogram(
            x=final_b,
            nbinsx=60,
            name=f"B: {scenario_b}",
            opacity=0.55,
            marker=dict(line=dict(width=1))
        ))
        fig.update_layout(barmode="overlay")

    fig.add_vline(
        x=median_a,
        line_dash="dash",
        annotation_text="Median (A)",
        annotation_position="top left",
    )

    if median_b is not None:
        fig.add_vline(
            x=median_b,
            line_dash="dash",
            annotation_text="Median (B)",
            annotation_position="top right",
        )

    if goal and goal > 0:
        fig.add_vline(
            x=goal,
            line_dash="dash",
            annotation_text="Goal",
            annotation_position="top",
        )

    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Final balance (real terms)",
        yaxis_title="Count",
        height=320,
    )
    return fig


def safe_now_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")


# -------------------- UI Components --------------------

def kpi_card(title: str, value_id: str):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="text-muted"),
            html.H3(id=value_id, className="mb-0"),
        ]),
        className="shadow-sm",
    )


navbar = dbc.Navbar(
    dbc.Container(
        dbc.Row(
            dbc.Col(html.H4("RiskWise", className="fw-bold text-center w-100 mb-0"), width=12),
            justify="center",
            className="w-100",
        ),
        fluid=True,
    ),
    color="dark",
    dark=True,
    className="mb-3",
)


# -------------------- Inputs Card --------------------

inputs_card = dbc.Card(
    dbc.CardBody([
        html.H5("Inputs", className="card-title"),

        html.Label("Life Goal (Preset)", className="mt-2"),
        dcc.Dropdown(
            options=[
                {"label": "Emergency Fund", "value": "emergency"},
                {"label": "House Deposit", "value": "deposit"},
                {"label": "Retirement", "value": "retirement"},
                {"label": "Custom", "value": "custom"},
            ],
            value="custom",
            id="preset",
            clearable=False,
            searchable=True,
        ),

        html.Label("Scenario", className="mt-3"),
        dcc.Dropdown(
            options=[{"label": k.title(), "value": k} for k in SCENARIOS.keys()],
            value="balanced",
            id="scenario",
            clearable=False,
        ),

        dbc.Checklist(
            options=[{"label": "Compare two scenarios", "value": "on"}],
            value=[],
            id="compare",
            switch=True,
            className="mt-3",
        ),

        html.Label("Scenario B (Comparison)", className="mt-2"),
        dcc.Dropdown(
            options=[{"label": k.title(), "value": k} for k in SCENARIOS.keys()],
            value="aggressive",
            id="scenario_b",
            clearable=False,
        ),

        html.Hr(),

        html.Label("Simulation quality", className="mt-1"),
        dbc.RadioItems(
            id="sim_quality",
            options=[
                {"label": "Fast", "value": "fast"},
                {"label": "Standard", "value": "standard"},
                {"label": "Accurate", "value": "accurate"},
            ],
            value="standard",
            inline=True,
        ),
        html.Div(id="sim-note", className="text-muted mt-1", style={"fontSize": "0.85rem"}),

        html.Hr(),

        dbc.Checklist(
            options=[{"label": "Use real market data (calibrate return/vol)", "value": "on"}],
            value=[],
            id="use_real_data",
            switch=True,
            className="mt-2",
        ),

        html.Label("Asset for calibration", className="mt-2"),
        dcc.Dropdown(
            options=[
                {"label": "S&P 500 (SPY)", "value": "spy.us"},
                {"label": "NASDAQ 100 (QQQ)", "value": "qqq.us"},
                {"label": "Russell 2000 (IWM)", "value": "iwm.us"},
            ],
            value="spy.us",
            id="asset",
            clearable=False,
        ),

        html.Div(id="data-note", className="text-muted mt-2", style={"fontSize": "0.9rem"}),

        html.Hr(),

        html.Label("Annual fees (%)", className="mt-2"),
        dcc.Slider(
            0.0, 2.0, 0.1,
            value=0.3,
            id="fees_pct",
            marks={0.0: "0%", 0.5: "0.5%", 1.0: "1%", 1.5: "1.5%", 2.0: "2%"},
        ),

        dbc.Checklist(
            options=[{"label": "Stress test: include a market crash", "value": "on"}],
            value=[],
            id="crash_on",
            switch=True,
            className="mt-3",
        ),

        html.Label("Crash severity (%)", className="mt-2"),
        dcc.Slider(
            10, 60, 5,
            value=30,
            id="crash_sev",
            marks={10: "10%", 30: "30%", 50: "50%", 60: "60%"},
        ),

        html.Hr(),

        html.Label("Years", className="mt-2"),
        dcc.Slider(
            1, 40, 1,
            value=10,
            id="years",
            marks={1: "1", 10: "10", 20: "20", 30: "30", 40: "40"},
        ),

        html.Label("Monthly Contribution", className="mt-3"),
        dcc.Slider(
            0, 2000, 50,
            value=200,
            id="monthly",
            marks={0: "0", 500: "500", 1000: "1000", 1500: "1500", 2000: "2000"},
        ),

        html.Label("Goal (£)", className="mt-3"),
        dbc.Input(type="number", value=50000, id="goal", min=0, step=1000),

        html.Hr(),

        html.H6("Goal planner (recommended saving)", className="mt-2"),
        html.Label("Target success probability (%)", className="mt-2"),
        dcc.Slider(
            50, 95, 5,
            value=75,
            id="target_prob",
            marks={50: "50%", 60: "60%", 75: "75%", 90: "90%", 95: "95%"},
        ),
        dbc.Button("Solve monthly contribution", id="solve-btn", color="primary", className="mt-2", n_clicks=0),
        html.Div(id="solver-result", className="text-muted mt-2", style={"fontSize": "0.95rem"}),

        html.Hr(),

        dbc.Checklist(
            options=[{"label": "Sensitivity: show goal probability if returns are ±1%", "value": "on"}],
            value=[],
            id="sens_on",
            switch=True,
            className="mt-2",
        ),
    ]),
    className="shadow-sm",
)


# -------------------- Summary & Insights --------------------

summary_card = dbc.Card(
    dbc.CardBody([
        html.H5("Summary & Insights", className="card-title"),
        dbc.Row([
            dbc.Col(dbc.Alert(id="insight-key", color="secondary"), width=12, className="mb-2"),
        ]),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Financial literacy insights"), html.Div(id="insight-literacy")])) ,
                    width=12, className="mb-2"),
        ]),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Deterministic vs Monte Carlo"), html.Div(id="insight-det")])) ,
                    width=12, className="mb-2"),
        ]),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Sensitivity"), html.Div(id="insight-sens")])) ,
                    width=12),
        ]),
    ]),
    className="shadow-sm",
)


# -------------------- Dashboard Tab --------------------

dashboard_tab = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col(kpi_card("Median outcome", "kpi-median"), width=3),
        dbc.Col(kpi_card("10th–90th range", "kpi-range"), width=3),
        dbc.Col(kpi_card("Goal probability", "kpi-goalprob"), width=3),
        dbc.Col(kpi_card("Assumptions used", "kpi-assumptions"), width=3),
    ], className="g-3 mb-3"),

    dbc.Row([
        dbc.Col(inputs_card, width=4),

        dbc.Col([
            dbc.Card(dbc.CardBody([
                html.H5("Wealth over time (Median + Uncertainty Band)", className="card-title"),
                dcc.Graph(id="paths-graph", config={"displayModeBar": False}),
            ]), className="shadow-sm mb-3"),

            dbc.Card(dbc.CardBody([
                html.H5("Distribution of final outcomes", className="card-title"),
                dcc.Graph(id="hist-graph", config={"displayModeBar": False}),
            ]), className="shadow-sm mb-3"),

            summary_card
        ], width=8),
    ], className="g-3")
])


# -------------------- My Plan Tab --------------------

def plan_help_panel():
    return dbc.Card(
        dbc.CardBody([
            html.H4("My Plan (for non-technical users)"),
            html.P(
                "This section turns the dashboard outputs into an actionable plan. "
                "Save a snapshot of your current settings + results, then review it later "
                "or download it (useful for a hackathon demo).",
                className="text-muted"
            ),
            html.Hr(),
            html.H6("Suggested workflow"),
            html.Ol([
                html.Li("Go to Dashboard → set your goal, years, and monthly contribution."),
                html.Li("Switch on Fees + Stress Test to understand realism and downside."),
                html.Li("Use Goal Planner to solve for a target success probability."),
                html.Li("Return here → Save Snapshot."),
                html.Li("Compare multiple snapshots: different scenarios, horizons, and contributions."),
            ]),
            dbc.Alert(
                "Tip: Save at least 3 snapshots (Cautious / Balanced / Aggressive) for a stronger presentation.",
                color="info",
                className="mt-2"
            ),
        ]),
        className="shadow-sm"
    )


myplan_tab = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col(plan_help_panel(), width=5),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Saved snapshots", className="card-title"),
            dbc.Button("Save current snapshot", id="save-plan-btn", color="primary", n_clicks=0, className="me-2"),
            dbc.Button("Clear all", id="clear-plans-btn", color="danger", outline=True, n_clicks=0),
            html.Hr(),

            # IMPORTANT: plan-select exists ALWAYS (no more dynamic id errors)
            dcc.Dropdown(
                id="plan-select",
                options=[],
                value=None,
                placeholder="No plans saved yet",
                clearable=False,
            ),
            html.Div(id="plan-list", className="mt-2"),
        ]), className="shadow-sm"), width=7),
    ], className="g-3"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Selected plan report", className="card-title"),
            html.Div(id="plan-report"),
            html.Hr(),
            dbc.Button("Download selected plan (JSON)", id="download-plan-btn", color="secondary", n_clicks=0),
            dcc.Download(id="download-plan"),
        ]), className="shadow-sm"), width=12),
    ], className="g-3 mt-1")
])


# -------------------- Sequence Demo Tab --------------------

sequence_tab = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("Sequence-of-Returns Risk (Interactive Demo)"),
            html.P(
                "Two investors can have the same average return, but different outcomes depending on when "
                "good/bad months happen. This is sequence-of-returns risk — especially important when "
                "you’re making regular contributions.",
                className="text-muted"
            ),
            dbc.Alert(
                "This demo uses the same set of monthly returns, but rearranges them so the worst months happen "
                "early vs late. Average return is the same — order is different.",
                color="warning"
            ),

            dbc.Row([
                dbc.Col([
                    html.Label("Years"),
                    dcc.Slider(1, 30, 1, value=10, id="seq-years",
                               marks={1: "1", 10: "10", 20: "20", 30: "30"}),
                ], width=4),
                dbc.Col([
                    html.Label("Monthly contribution (£)"),
                    dcc.Slider(0, 2000, 50, value=200, id="seq-monthly",
                               marks={0: "0", 500: "500", 1000: "1000", 1500: "1500", 2000: "2000"}),
                ], width=4),
                dbc.Col([
                    html.Label("Avg annual return (%)"),
                    dcc.Slider(0, 15, 0.5, value=7.0, id="seq-avg",
                               marks={0: "0%", 5: "5%", 10: "10%", 15: "15%"}),
                ], width=4),
            ], className="g-3"),

            dbc.Row([
                dbc.Col([
                    html.Label("Annual volatility (%)"),
                    dcc.Slider(1, 40, 1, value=15, id="seq-vol",
                               marks={5: "5%", 15: "15%", 25: "25%", 40: "40%"}),
                ], width=4),
                dbc.Col([
                    html.Label("Random seed"),
                    dbc.Input(type="number", value=7, id="seq-seed", min=1, step=1),
                    html.Div("Change seed to generate a different return set.", className="text-muted", style={"fontSize": "0.85rem"}),
                ], width=4),
                dbc.Col([
                    html.Label("Start balance (£)"),
                    dbc.Input(type="number", value=1000, id="seq-start", min=0, step=100),
                ], width=4),
            ], className="g-3 mt-1"),

        ]), className="shadow-sm"), width=12),
    ]),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Outcome depends on *order* (not just average)", className="card-title"),
            dcc.Graph(id="seq-graph", config={"displayModeBar": False}),
            html.Div(id="seq-summary", className="mt-2"),
        ]), className="shadow-sm"), width=12),
    ], className="g-3 mt-2")
])


# -------------------- Learn & Method Tab --------------------

learn_tab = dbc.Container(fluid=True, className="pt-2", children=[
    dbc.Row([
        dbc.Col([
            dbc.Card(dbc.CardBody([
                html.H4("Learn & Method"),
                html.P(
                    "RiskWise is built to teach probabilistic thinking in personal finance. "
                    "It doesn’t predict markets — it illustrates uncertainty.",
                    className="text-muted"
                ),

                html.Hr(),
                html.H5("1) What RiskWise is doing (high level)"),
                html.Ul([
                    html.Li("You choose assumptions (return, volatility, fees, inflation, horizon, contributions)."),
                    html.Li("RiskWise generates thousands of plausible future paths using Monte Carlo simulation."),
                    html.Li("It summarises the distribution: median, 10th/90th percentiles, goal probability."),
                    html.Li("It also shows a deterministic projection (one ‘average’ path) so you can see why a single line is misleading."),
                ]),

                html.Hr(),
                html.H5("2) Monte Carlo simulation (intuition)"),
                html.P(
                    "Monte Carlo simulation runs many random trials. In investing terms: instead of one future, you simulate many "
                    "possible sequences of monthly returns. The distribution of outcomes tells you typical ranges and tail risk.",
                    className="text-muted"
                ),
                html.Ul([
                    html.Li("Median ≈ best ‘typical’ outcome (50th percentile)."),
                    html.Li("10th–90th ≈ a practical uncertainty band (not worst-case, not best-case)."),
                    html.Li("Goal probability = fraction of simulations that finish above your target goal."),
                ]),

                html.Hr(),
                html.H5("3) Real market calibration (SPY / QQQ / IWM)"),
                html.P(
                    "If enabled, RiskWise downloads historical prices and estimates annual return/volatility using log returns, "
                    "then annualises the estimates. This gives a more data-driven baseline than a fixed preset.",
                    className="text-muted"
                ),

                html.Hr(),
                html.H5("4) Fees, inflation, and why ‘real’ terms matter"),
                html.Ul([
                    html.Li("Fees reduce expected return each year — even small fees compound over time."),
                    html.Li("Inflation reduces purchasing power — RiskWise reports values in ‘today’s money’ (real terms)."),
                ]),

                html.Hr(),
                html.H5("5) Stress testing (crash scenario)"),
                html.P(
                    "The stress test applies a one-off crash to simulated paths. It’s simplified, but helps communicate downside "
                    "risk and why investors need margin for error.",
                    className="text-muted"
                ),

                html.Hr(),
                html.H5("6) Deterministic vs Monte Carlo (why we show both)"),
                html.Ul([
                    html.Li("Deterministic forecast = one line, assumes the average happens smoothly."),
                    html.Li("Monte Carlo = range of outcomes, shows dispersion and goal probability."),
                    html.Li("Seeing both helps users understand why ‘one number’ is not reality."),
                ]),

                html.Hr(),
                html.H5("7) Goal planner (binary search solver)"),
                html.P(
                    "The solver searches for the minimum monthly contribution needed to reach your goal with a chosen probability "
                    "(e.g. 75%). It uses binary search for speed and stability.",
                    className="text-muted"
                ),

                html.Hr(),
                html.H5("Limitations (important)"),
                html.Ul([
                    html.Li("Educational tool, not financial advice."),
                    html.Li("Historical patterns may not repeat."),
                    html.Li("Volatility and returns are simplified."),
                    html.Li("Crash model is illustrative (not predictive)."),
                ]),
            ]), className="shadow-sm")
        ], width=10),
    ], justify="center")
])


# -------------------- Onboarding Modal --------------------

onboarding_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Welcome to RiskWise")),
        dbc.ModalBody([
            html.P(
                "RiskWise visualises uncertainty in long-term investing. Instead of one forecast, "
                "it shows a range of plausible outcomes.",
                className="text-muted"
            ),
            html.Ul([
                html.Li("Median: best-guess typical outcome"),
                html.Li("Band: typical uncertainty range"),
                html.Li("Goal probability: chance of success under assumptions"),
            ]),
            dbc.Alert("Educational tool, not financial advice.", color="warning", className="mt-2"),
        ]),
        dbc.ModalFooter(
            dbc.Button("Continue →", id="continue-btn", color="primary", n_clicks=0)
        ),
    ],
    id="welcome-modal",
    is_open=True,
    backdrop="static",
    centered=True,
)


# -------------------- Layout --------------------

app.layout = dbc.Container(fluid=True, children=[
    navbar,

    # Stores MUST always exist in layout to avoid "nonexistent object" errors
    dcc.Store(id="onboarded", storage_type="session", data=False),
    dcc.Store(id="plans-store", storage_type="local", data={"plans": [], "selected_index": None}),
    dcc.Store(id="dash-snapshot", storage_type="memory", data=None),
    dcc.Store(id="solver-store", storage_type="memory", data=""),

    onboarding_modal,

    dbc.Tabs(
        [
            dbc.Tab(label="Dashboard", tab_id="tab-dashboard"),
            dbc.Tab(label="My Plan", tab_id="tab-plan"),
            dbc.Tab(label="Sequence Demo", tab_id="tab-seq"),
            dbc.Tab(label="Learn & Method", tab_id="tab-learn"),
        ],
        id="tabs",
        active_tab="tab-dashboard",
    ),

    html.Div(id="tab-content", className="pt-3"),
])


# -------------------- Callbacks --------------------

@app.callback(
    Output("welcome-modal", "is_open"),
    Output("onboarded", "data"),
    Input("continue-btn", "n_clicks"),
    State("onboarded", "data"),
    prevent_initial_call=True
)
def close_modal(n, onboarded):
    return False, True


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
)
def render_tab(active_tab):
    if active_tab == "tab-plan":
        return myplan_tab
    if active_tab == "tab-seq":
        return sequence_tab
    if active_tab == "tab-learn":
        return learn_tab
    return dashboard_tab


@app.callback(
    Output("years", "value"),
    Output("monthly", "value"),
    Output("goal", "value"),
    Input("preset", "value"),
)
def apply_preset(preset):
    return PRESETS.get(preset, PRESETS["custom"])


# -------------------- Cache current dashboard inputs into a Store --------------------
# This is the key fix: My Plan uses dash-snapshot instead of State("scenario", ...)

@app.callback(
    Output("dash-snapshot", "data"),
    Input("scenario", "value"),
    Input("scenario_b", "value"),
    Input("compare", "value"),
    Input("sim_quality", "value"),
    Input("years", "value"),
    Input("monthly", "value"),
    Input("goal", "value"),
    Input("use_real_data", "value"),
    Input("asset", "value"),
    Input("fees_pct", "value"),
    Input("crash_on", "value"),
    Input("crash_sev", "value"),
    Input("target_prob", "value"),
    Input("sens_on", "value"),
)
def snapshot_cache(
    scenario, scenario_b, compare, sim_quality,
    years, monthly, goal, use_real_data, asset,
    fees_pct, crash_on, crash_sev, target_prob, sens_on
):
    return {
        "scenario": scenario,
        "scenario_b": scenario_b,
        "compare": (compare or []),
        "sim_quality": sim_quality,
        "years": years,
        "monthly": monthly,
        "goal": goal,
        "use_real_data": (use_real_data or []),
        "asset": asset,
        "fees_pct": fees_pct,
        "crash_on": (crash_on or []),
        "crash_sev": crash_sev,
        "target_prob": target_prob,
        "sens_on": (sens_on or []),
        "cached_at": safe_now_str(),
    }


# -------------------- Dashboard engine --------------------

@app.callback(
    Output("paths-graph", "figure"),
    Output("hist-graph", "figure"),
    Output("data-note", "children"),
    Output("kpi-median", "children"),
    Output("kpi-range", "children"),
    Output("kpi-goalprob", "children"),
    Output("kpi-assumptions", "children"),
    Output("sim-note", "children"),
    Output("insight-key", "children"),
    Output("insight-literacy", "children"),
    Output("insight-det", "children"),
    Output("insight-sens", "children"),
    Input("scenario", "value"),
    Input("scenario_b", "value"),
    Input("compare", "value"),
    Input("sim_quality", "value"),
    Input("years", "value"),
    Input("monthly", "value"),
    Input("goal", "value"),
    Input("use_real_data", "value"),
    Input("asset", "value"),
    Input("fees_pct", "value"),
    Input("crash_on", "value"),
    Input("crash_sev", "value"),
    Input("sens_on", "value"),
)
def update_dashboard(
    scenario, scenario_b, compare, sim_quality,
    years, monthly, goal,
    use_real_data, asset,
    fees_pct, crash_on, crash_sev,
    sens_on
):
    compare_on = "on" in (compare or [])
    use_data = "on" in (use_real_data or [])
    sens_enabled = "on" in (sens_on or [])

    q = SIM_QUALITY.get(sim_quality, SIM_QUALITY["standard"])
    sims = int(q["sims"])
    seed_a = int(q["seed_a"])
    seed_b = int(q["seed_b"])
    sim_note = f"Runs {sims:,} simulations per scenario (higher = smoother, slower)."

    crash_enabled = "on" in (crash_on or [])
    crash_pct = (float(crash_sev) / 100.0) if crash_enabled else 0.0
    fee_pct = float(fees_pct) / 100.0

    params_a = SCENARIOS[scenario]
    params_b = SCENARIOS[scenario_b]

    data_note = "Using preset assumptions"

    def get_r_vol(default_params):
        nonlocal data_note
        if not use_data:
            return default_params["r"], default_params["vol"]

        try:
            close = fetch_stooq_close_prices(asset)
            mu, vol = estimate_annual_return_vol(close)
            mu = float(np.clip(mu, -0.05, 0.20))
            vol = float(np.clip(vol, 0.05, 0.60))
            data_note = f"Calibrated from {asset.upper()} historical data"
            return mu, vol
        except Exception:
            data_note = "Calibration failed (network/data). Using presets."
            return default_params["r"], default_params["vol"]

    r_a, v_a = get_r_vol(params_a)
    r_a_net = max(-0.50, r_a - fee_pct)

    paths_a = simulate(
        years=years,
        start_balance=1000,
        monthly_contribution=monthly,
        expected_return_annual=r_a_net,
        volatility_annual=v_a,
        inflation_annual=INFLATION_ANNUAL,
        simulations=sims,
        seed=seed_a
    )
    if crash_enabled:
        paths_a = apply_crash(paths_a, crash_pct=crash_pct, seed=1001)

    final_a = paths_a[:, -1]
    summary_a = summarise(final_a)
    p_goal_a = prob_reach_goal(final_a, goal if goal else 0)

    det_a = deterministic_forecast(
        years=years,
        start_balance=1000,
        monthly_contribution=monthly,
        expected_return_annual=r_a,
        inflation_annual=INFLATION_ANNUAL,
        fee_annual=fee_pct
    )

    paths_b = final_b = summary_b = None
    if compare_on:
        if scenario_b == scenario:
            paths_b = paths_a
            final_b = final_a
            summary_b = summary_a
        else:
            r_b, v_b = get_r_vol(params_b)
            r_b_net = max(-0.50, r_b - fee_pct)

            paths_b = simulate(
                years=years,
                start_balance=1000,
                monthly_contribution=monthly,
                expected_return_annual=r_b_net,
                volatility_annual=v_b,
                inflation_annual=INFLATION_ANNUAL,
                simulations=sims,
                seed=seed_b
            )
            if crash_enabled:
                paths_b = apply_crash(paths_b, crash_pct=crash_pct, seed=1002)

            final_b = paths_b[:, -1]
            summary_b = summarise(final_b)

    p10_a, p50_a, p90_a = pct_band(paths_a)
    months = np.arange(len(p50_a))

    p10_b = p50_b = p90_b = None
    b_name = None
    if compare_on and paths_b is not None and scenario_b != scenario:
        p10_b, p50_b, p90_b = pct_band(paths_b)
        b_name = scenario_b.title()

    fig_paths = build_paths_figure(
        months=months,
        scenario_name=scenario.title(),
        p10=p10_a, p50=p50_a, p90=p90_a,
        det=det_a,
        p10_b=p10_b, p50_b=p50_b, p90_b=p90_b,
        scenario_b_name=b_name
    )

    fig_hist = build_hist_figure(
        final_a=final_a,
        scenario_a=scenario.title(),
        median_a=summary_a["median"],
        goal=goal,
        final_b=final_b if (compare_on and final_b is not None and scenario_b != scenario) else None,
        scenario_b=scenario_b.title() if (compare_on and final_b is not None and scenario_b != scenario) else None,
        median_b=summary_b["median"] if (compare_on and summary_b is not None and scenario_b != scenario) else None
    )

    kpi_median = money(summary_a["median"])
    kpi_range = f"{money(summary_a['p10'])} – {money(summary_a['p90'])}"
    kpi_goalprob = f"{p_goal_a*100:.1f}%" if goal and goal > 0 else "—"
    kpi_assumptions = f"Net return {r_a_net*100:.1f}% • Vol {v_a*100:.1f}% • Infl {INFLATION_ANNUAL*100:.1f}%"

    spread_a = summary_a["p90"] - summary_a["p10"]
    det_end = float(det_a[-1]) if len(det_a) else float("nan")

    if goal and goal > 0:
        if p_goal_a >= 0.75:
            status = "✅ On track"
            action = "Keep saving and consider stress-testing to understand downside."
        elif p_goal_a >= 0.40:
            status = "🟡 Borderline"
            action = "Consider a slightly higher monthly saving or a longer horizon."
        else:
            status = "🔴 Unlikely"
            action = "Increase monthly saving, extend time horizon, or reduce goal."
        key_text = (
            f"{status}: median outcome is {money(summary_a['median'])}. "
            f"Typical range is {money(summary_a['p10'])}–{money(summary_a['p90'])}. "
            f"Goal probability is {p_goal_a*100:.1f}%. {action}"
        )
    else:
        key_text = (
            f"Median outcome is {money(summary_a['median'])} with a typical range "
            f"{money(summary_a['p10'])}–{money(summary_a['p90'])}. "
            f"Add a goal to calculate success probability."
        )

    fee_line = f"Fees: {fees_pct:.2f}%/yr reduces expected growth; small % fees compound over time."
    infl_line = f"Inflation: results are shown in real terms (today’s money) using {INFLATION_ANNUAL*100:.1f}%/yr."
    unc_line = f"Uncertainty: the 10th–90th band span is {money(spread_a)} (realistic spread, not a prediction)."
    crash_line = f"Stress test: {'ON' if crash_enabled else 'OFF'}" + (f" (crash {crash_sev:.0f}%)." if crash_enabled else ".")
    calib_line = f"Calibration: {'real market data' if use_data else 'preset assumptions'}."

    literacy = html.Ul([
        html.Li(fee_line),
        html.Li(infl_line),
        html.Li(unc_line),
        html.Li(crash_line),
        html.Li(calib_line),
    ], className="mb-0")

    det_vs_mc = html.Div([
        html.P(f"Deterministic forecast (single average path): {money(det_end)}", className="mb-1"),
        html.P(f"Monte Carlo median (typical outcome): {money(summary_a['median'])}", className="mb-1"),
        html.Small("Deterministic projections hide dispersion. Monte Carlo shows ranges and probability of success.", className="text-muted"),
    ])

    if sens_enabled and goal and goal > 0:
        sens_sims = 900

        def run_prob(delta):
            paths = simulate(
                years=years,
                start_balance=1000,
                monthly_contribution=monthly,
                expected_return_annual=max(-0.50, (r_a_net + delta)),
                volatility_annual=v_a,
                inflation_annual=INFLATION_ANNUAL,
                simulations=sens_sims,
                seed=777 + int((delta + 1) * 1000)
            )
            if crash_enabled:
                paths = apply_crash(paths, crash_pct=crash_pct, seed=3000 + int((delta + 1) * 1000))
            return prob_reach_goal(paths[:, -1], goal)

        p_low = run_prob(-0.01)
        p_high = run_prob(+0.01)
        sens_block = html.Ul([
            html.Li(f"1% lower net return → goal probability {p_low*100:.1f}%"),
            html.Li(f"Current net return → goal probability {p_goal_a*100:.1f}%"),
            html.Li(f"1% higher net return → goal probability {p_high*100:.1f}%"),
        ], className="mb-0")
    else:
        sens_block = html.Div(
            "Enable sensitivity to see how much your goal probability changes if returns are ±1%.",
            className="text-muted"
        )

    return (
        fig_paths, fig_hist,
        data_note,
        kpi_median, kpi_range, kpi_goalprob, kpi_assumptions,
        sim_note,
        key_text,
        literacy,
        det_vs_mc,
        sens_block
    )


# -------------------- Goal Planner solver (also cache text in solver-store) --------------------

@app.callback(
    Output("solver-result", "children"),
    Output("solver-store", "data"),
    Input("solve-btn", "n_clicks"),
    State("scenario", "value"),
    State("years", "value"),
    State("goal", "value"),
    State("target_prob", "value"),
    State("use_real_data", "value"),
    State("asset", "value"),
    State("fees_pct", "value"),
    State("crash_on", "value"),
    State("crash_sev", "value"),
    prevent_initial_call=True
)
def solve_monthly(
    n_clicks, scenario, years, goal, target_prob,
    use_real_data, asset, fees_pct, crash_on, crash_sev
):
    if not goal or goal <= 0:
        msg = "Set a Goal (£) above 0, then click Solve."
        return msg, msg

    use_data = "on" in (use_real_data or [])
    crash_enabled = "on" in (crash_on or [])
    crash_pct = (float(crash_sev) / 100.0) if crash_enabled else 0.0
    fee_pct = float(fees_pct) / 100.0
    target = float(target_prob) / 100.0

    default_params = SCENARIOS[scenario]

    def get_r_vol():
        if not use_data:
            return default_params["r"], default_params["vol"]
        try:
            close = fetch_stooq_close_prices(asset)
            mu, vol = estimate_annual_return_vol(close)
            mu = float(np.clip(mu, -0.05, 0.20))
            vol = float(np.clip(vol, 0.05, 0.60))
            return mu, vol
        except Exception:
            return default_params["r"], default_params["vol"]

    r, v = get_r_vol()
    r_net = max(-0.50, r - fee_pct)

    lo, hi = 0.0, 5000.0
    sims_solver = 900
    iters = 12

    for _ in range(iters):
        mid = (lo + hi) / 2.0

        paths = simulate(
            years=years,
            start_balance=1000,
            monthly_contribution=mid,
            expected_return_annual=r_net,
            volatility_annual=v,
            inflation_annual=INFLATION_ANNUAL,
            simulations=sims_solver,
            seed=123
        )
        if crash_enabled:
            paths = apply_crash(paths, crash_pct=crash_pct, seed=2001)

        p = prob_reach_goal(paths[:, -1], goal)

        if p >= target:
            hi = mid
        else:
            lo = mid

    recommended = round(hi / 10) * 10
    msg = f"To reach {money(goal)} with ~{int(target_prob)}% probability, save about {money(recommended)}/month (under current settings)."
    return msg, msg


# -------------------- My Plan: store snapshots (FIXED to use dash-snapshot + solver-store) --------------------

@app.callback(
    Output("plans-store", "data"),
    Input("save-plan-btn", "n_clicks"),
    Input("clear-plans-btn", "n_clicks"),
    State("plans-store", "data"),
    State("dash-snapshot", "data"),
    State("solver-store", "data"),
    prevent_initial_call=True
)
def plans_store_update(save_clicks, clear_clicks, store, snap, solver_text):
    store = store or {"plans": [], "selected_index": None}

    import dash
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

    if triggered == "clear-plans-btn":
        return {"plans": [], "selected_index": None}

    if triggered != "save-plan-btn":
        return store

    if not snap:
        # user hasn't visited dashboard yet / no cached inputs
        plan = {
            "saved_at": safe_now_str(),
            "inputs": {},
            "solver_result": str(solver_text or ""),
            "note": "No dashboard snapshot found yet. Visit Dashboard first."
        }
    else:
        compare_on = "on" in (snap.get("compare") or [])
        use_data = "on" in (snap.get("use_real_data") or [])
        crash_enabled = "on" in (snap.get("crash_on") or [])

        plan = {
            "saved_at": safe_now_str(),
            "inputs": {
                "scenario": snap.get("scenario"),
                "compare_on": compare_on,
                "scenario_b": snap.get("scenario_b"),
                "years": int(snap.get("years") or 0),
                "monthly": float(snap.get("monthly") or 0.0),
                "goal": float(snap.get("goal") or 0.0),
                "use_real_data": use_data,
                "asset": snap.get("asset"),
                "fees_pct": float(snap.get("fees_pct") or 0.0),
                "crash_enabled": crash_enabled,
                "crash_sev": float(snap.get("crash_sev") or 0.0),
                "target_prob": float(snap.get("target_prob") or 0.0),
                "sim_quality": snap.get("sim_quality"),
            },
            "solver_result": str(solver_text) if solver_text is not None else "",
            "cached_at": snap.get("cached_at"),
        }

    plans = list(store.get("plans", []))
    plans.insert(0, plan)
    return {"plans": plans, "selected_index": 0}


@app.callback(
    Output("plan-select", "options"),
    Output("plan-select", "value"),
    Output("plan-list", "children"),
    Output("plan-report", "children"),
    Input("plans-store", "data"),
    Input("plan-select", "value"),
)
def render_plans(store, selected_idx):
    store = store or {"plans": [], "selected_index": None}
    plans = store.get("plans", [])

    if not plans:
        return [], None, dbc.Alert(
            "No saved plans yet. Go to Dashboard → set inputs → come back and click 'Save current snapshot'.",
            color="info"
        ), dbc.Alert("No plan selected.", color="secondary")

    if selected_idx is None:
        selected_idx = store.get("selected_index", 0)
    try:
        selected_idx = int(selected_idx)
    except Exception:
        selected_idx = 0
    selected_idx = max(0, min(selected_idx, len(plans) - 1))

    options = [
        {
            "label": f"{i}: {p.get('saved_at','')} • "
                     f"{(p.get('inputs',{}).get('scenario') or '—').title()} • "
                     f"{money(p.get('inputs',{}).get('monthly',0))}/mo • "
                     f"Goal {money(p.get('inputs',{}).get('goal',0))}",
            "value": i
        }
        for i, p in enumerate(plans)
    ]

    p = plans[selected_idx]
    inp = p.get("inputs", {})

    list_ui = dbc.Alert(
        "Select a snapshot above to view the full plan report below.",
        color="secondary"
    )

    scenario = (inp.get("scenario") or "—").title()
    scenario_b = (inp.get("scenario_b") or "—").title()

    report = dbc.Card(dbc.CardBody([
        html.H5("Plan report"),
        html.Small(f"Saved: {p.get('saved_at','')}", className="text-muted"),
        html.Hr(),

        html.H6("Your settings"),
        html.Ul([
            html.Li(f"Scenario: {scenario}" + (f" (comparison vs {scenario_b})" if inp.get("compare_on") else "")),
            html.Li(f"Time horizon: {inp.get('years','—')} years"),
            html.Li(f"Monthly contribution: {money(inp.get('monthly',0))}"),
            html.Li(f"Goal: {money(inp.get('goal',0))}"),
            html.Li(f"Fees: {float(inp.get('fees_pct',0.0)):.2f}%/yr"),
            html.Li(
                f"Stress test: {'ON' if inp.get('crash_enabled') else 'OFF'}" +
                (f" ({float(inp.get('crash_sev',0.0)):.0f}% crash)" if inp.get("crash_enabled") else "")
            ),
            html.Li(
                f"Calibration: {'real market data' if inp.get('use_real_data') else 'preset'}" +
                (f" ({inp.get('asset')})" if inp.get("use_real_data") else "")
            ),
            html.Li(f"Simulation quality: {str(inp.get('sim_quality','standard')).title()}"),
        ]),

        html.H6("Recommended action"),
        dbc.Alert(
            p.get("solver_result", "") or "Use the Goal Planner on Dashboard to compute a recommended monthly saving.",
            color="info"
        ),

        html.H6("Plain-English next steps"),
        html.Ol([
            html.Li("If probability is low, increase monthly saving or extend the time horizon."),
            html.Li("Stress-test your plan to see realistic downside outcomes."),
            html.Li("Compare cautious vs aggressive scenarios and pick a plan you can stick to."),
        ]),
    ]), className="shadow-sm")

    return options, selected_idx, list_ui, report


@app.callback(
    Output("download-plan", "data"),
    Input("download-plan-btn", "n_clicks"),
    State("plans-store", "data"),
    State("plan-select", "value"),
    prevent_initial_call=True
)
def download_selected_plan(n, store, selected_idx):
    store = store or {"plans": [], "selected_index": None}
    plans = store.get("plans", [])
    if not plans:
        return None

    try:
        i = int(selected_idx) if selected_idx is not None else 0
    except Exception:
        i = 0
    i = max(0, min(i, len(plans) - 1))
    payload = plans[i]

    filename = f"riskwise_plan_{i}_{payload.get('saved_at','').replace(':','-').replace(' ','_')}.json"
    return dict(content=json.dumps(payload, indent=2), filename=filename)


# -------------------- Sequence Demo --------------------

def sequence_paths(
    years: int,
    start_balance: float,
    monthly_contribution: float,
    avg_annual: float,
    vol_annual: float,
    seed: int
):
    months = years * 12
    rng = np.random.default_rng(seed)

    mu_m = (1.0 + avg_annual) ** (1.0 / 12.0) - 1.0
    vol_m = vol_annual / np.sqrt(12.0)

    rets = rng.normal(loc=mu_m, scale=vol_m, size=months)
    rets_sorted = np.sort(rets)

    rets_worst_early = rets_sorted.copy()
    rets_worst_late = rets_sorted[::-1].copy()

    infl_m = (1.0 + INFLATION_ANNUAL) ** (1.0 / 12.0) - 1.0

    def apply_returns(rseq):
        bal = start_balance
        out = np.zeros(months + 1)
        out[0] = bal
        for t in range(1, months + 1):
            bal = bal * (1.0 + rseq[t - 1]) + monthly_contribution
            out[t] = bal / ((1.0 + infl_m) ** t)
        return out

    a = apply_returns(rets_worst_early)
    b = apply_returns(rets_worst_late)

    return a, b, rets.mean()


@app.callback(
    Output("seq-graph", "figure"),
    Output("seq-summary", "children"),
    Input("seq-years", "value"),
    Input("seq-monthly", "value"),
    Input("seq-avg", "value"),
    Input("seq-vol", "value"),
    Input("seq-seed", "value"),
    Input("seq-start", "value"),
)
def update_sequence_demo(years, monthly, avg_pct, vol_pct, seed, start):
    years = int(years)
    monthly = float(monthly)
    avg = float(avg_pct) / 100.0
    vol = float(vol_pct) / 100.0
    seed = int(seed) if seed is not None else 7
    start = float(start) if start is not None else 1000.0

    a, b, mean_m = sequence_paths(years, start, monthly, avg, vol, seed)
    months = np.arange(len(a))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=a, mode="lines", name="Worst returns early"))
    fig.add_trace(go.Scatter(x=months, y=b, mode="lines", name="Worst returns late"))

    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Months",
        yaxis_title="Balance (real terms)",
        height=420,
        legend_title="Scenario",
    )

    end_a = a[-1]
    end_b = b[-1]

    summary = dbc.Alert(
        f"Same average monthly return set (mean ≈ {mean_m*100:.2f}%/month), different order. "
        f"Final balance differs: worst-early {money(end_a)} vs worst-late {money(end_b)}. "
        f"This illustrates why volatility + timing matter, not just averages.",
        color="info"
    )
    return fig, summary


# -------------------- Main --------------------

if __name__ == "__main__":
    app.run(debug=True)
