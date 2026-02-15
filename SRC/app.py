from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

from simulator import simulate
from metrics import summarise, prob_reach_goal

# -------------------- Safe market_data import (prevents Render crashes) --------------------
try:
    from market_data import fetch_stooq_close_prices, estimate_annual_return_vol
    MARKET_DATA_AVAILABLE = True
except Exception:
    MARKET_DATA_AVAILABLE = False


# -------------------- Config --------------------
SCENARIOS = {
    "cautious":   {"r": 0.05, "vol": 0.10},
    "balanced":   {"r": 0.07, "vol": 0.15},
    "aggressive": {"r": 0.09, "vol": 0.20},
}

# Tuple order is (years, monthly, goal)
PRESETS = {
    "emergency":  (2,  300,  3000),
    "deposit":    (5,  500,  20000),
    "retirement": (30, 400,  500000),
    "custom":     (10, 200,  50000),
}


def money(x: float) -> str:
    return f"£{x:,.0f}"


def pct_band(paths: np.ndarray):
    p10 = np.percentile(paths, 10, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p90 = np.percentile(paths, 90, axis=0)
    return p10, p50, p90


def apply_crash(paths: np.ndarray, crash_pct: float, seed: int = 999) -> np.ndarray:
    """
    Stress-test: for each simulated path, pick a random crash month and
    apply a one-off drawdown of crash_pct to all future balances.
    Educational approximation to illustrate downside risk.
    """
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


def deterministic_path(
    years: int,
    start_balance: float,
    monthly_contribution: float,
    expected_return_annual: float,
    inflation_annual: float,
):
    """
    Deterministic "calculator" projection:
    single smooth line using constant return every month.
    Includes inflation adjustment by discounting to real terms.
    """
    n_months = int(years * 12)
    r_m = (1.0 + expected_return_annual) ** (1.0 / 12.0) - 1.0
    inf_m = (1.0 + inflation_annual) ** (1.0 / 12.0) - 1.0

    balances = np.zeros(n_months + 1, dtype=float)
    balances[0] = start_balance

    for t in range(1, n_months + 1):
        balances[t] = balances[t - 1] * (1.0 + r_m) + monthly_contribution

    # convert to real terms (today's money)
    discount = (1.0 + inf_m) ** np.arange(n_months + 1)
    real_balances = balances / discount
    return real_balances


def classify_probability(p: float) -> str:
    if p >= 0.75:
        return "high"
    if p >= 0.40:
        return "medium"
    return "low"


def plain_english_summary(goal: float, p_goal: float, p10: float, p90: float) -> str:
    level = classify_probability(p_goal)
    if goal <= 0:
        return "Set a goal above £0 to see a clear plan and probability."
    if level == "high":
        return (
            f"You're broadly on track. Under these assumptions, you have a {p_goal*100:.1f}% chance of reaching "
            f"{money(goal)}. Most outcomes fall between {money(p10)} and {money(p90)}."
        )
    if level == "medium":
        return (
            f"Your goal is achievable, but not guaranteed. Under these assumptions, you have a {p_goal*100:.1f}% chance "
            f"of reaching {money(goal)}. Most outcomes fall between {money(p10)} and {money(p90)}."
        )
    return (
        f"Your goal is unlikely under current settings. Under these assumptions, you have a {p_goal*100:.1f}% chance "
        f"of reaching {money(goal)}. Most outcomes fall between {money(p10)} and {money(p90)}."
    )


def recommended_actions(p_goal: float) -> list[str]:
    level = classify_probability(p_goal)
    if level == "high":
        return [
            "Keep contributions consistent — consistency matters more than optimisation.",
            "Consider lowering fees where possible (fees compound over time).",
            "Stress test with a crash to understand downside risk."
        ]
    if level == "medium":
        return [
            "Increase monthly contributions OR extend the time horizon to improve probability.",
            "Reduce fees if possible (fees compound).",
            "Try a less aggressive scenario to see stability trade-offs."
        ]
    return [
        "Increase monthly contributions (highest leverage).",
        "Extend the time horizon (time is the second biggest lever).",
        "Reduce fees and avoid overly aggressive assumptions."
    ]


# -------------------- UI components --------------------
def kpi_card(title: str, value_id: str):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="text-muted"),
            html.H3(id=value_id, className="mb-0")
        ]),
        className="shadow-sm"
    )


navbar = dbc.Navbar(
    dbc.Container(
        dbc.Row(
            dbc.Col(
                html.H4("RiskWise", className="fw-bold text-center w-100 mb-0"),
                width=12
            ),
            justify="center",
            className="w-100"
        ),
        fluid=True
    ),
    color="dark",
    dark=True,
    className="mb-3"
)


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
            clearable=False
        ),

        html.Label("Scenario", className="mt-3"),
        dcc.Dropdown(
            options=[{"label": k.title(), "value": k} for k in SCENARIOS.keys()],
            value="balanced",
            id="scenario",
            clearable=False
        ),

        dbc.Checklist(
            options=[{"label": "Compare two scenarios", "value": "on"}],
            value=[],
            id="compare",
            switch=True,
            className="mt-3"
        ),

        html.Div([
            html.Label("Scenario B (Comparison)", className="mt-2"),
            dcc.Dropdown(
                options=[{"label": k.title(), "value": k} for k in SCENARIOS.keys()],
                value="aggressive",
                id="scenario_b",
                clearable=False
            ),
        ], id="scenario-b-wrap"),

        html.Hr(),

        # NEW: deterministic toggle
        dbc.Checklist(
            options=[{"label": "Show deterministic (single-line) forecast", "value": "on"}],
            value=["on"],
            id="deterministic_on",
            switch=True,
            className="mt-2"
        ),

        html.Hr(),

        dbc.Checklist(
            options=[{"label": "Use real market data (calibrate return/vol)", "value": "on"}],
            value=[],
            id="use_real_data",
            switch=True,
            className="mt-2"
        ),

        html.Div([
            html.Label("Asset for calibration", className="mt-2"),
            dcc.Dropdown(
                options=[
                    {"label": "S&P 500 (SPY)", "value": "spy.us"},
                    {"label": "NASDAQ 100 (QQQ)", "value": "qqq.us"},
                    {"label": "Russell 2000 (IWM)", "value": "iwm.us"},
                ],
                value="spy.us",
                id="asset",
                clearable=False
            ),
            html.Div(id="data-note", className="text-muted mt-2", style={"fontSize": "0.9rem"}),
        ], id="asset-wrap"),

        html.Div(
            "Market calibration unavailable (market_data.py import failed). Using preset scenario assumptions.",
            id="market-missing-note",
            className="text-muted mt-2",
            style={"fontSize": "0.9rem", "display": "none"}
        ),

        html.Hr(),

        html.Label("Annual fees (%)", className="mt-2"),
        dcc.Slider(
            0.0, 2.0, 0.1,
            value=0.3,
            id="fees_pct",
            marks={0.0: "0%", 0.5: "0.5%", 1.0: "1%", 1.5: "1.5%", 2.0: "2%"}
        ),

        dbc.Checklist(
            options=[{"label": "Stress test: include a market crash", "value": "on"}],
            value=[],
            id="crash_on",
            switch=True,
            className="mt-3"
        ),

        html.Div([
            html.Label("Crash severity (%)", className="mt-2"),
            dcc.Slider(
                10, 60, 5,
                value=30,
                id="crash_sev",
                marks={10: "10%", 30: "30%", 50: "50%", 60: "60%"}
            ),
        ], id="crash-wrap"),

        html.Hr(),

        html.Label("Years", className="mt-2"),
        dcc.Slider(
            1, 40, 1, value=10, id="years",
            marks={1: "1", 10: "10", 20: "20", 30: "30", 40: "40"}
        ),

        html.Label("Monthly Contribution", className="mt-3"),
        dcc.Slider(
            0, 2000, 50, value=200, id="monthly",
            marks={0: "0", 500: "500", 1000: "1000", 1500: "1500", 2000: "2000"}
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
            marks={50: "50%", 60: "60%", 75: "75%", 90: "90%", 95: "95%"}
        ),
        dbc.Button("Solve monthly contribution", id="solve-btn", color="primary", className="mt-2", n_clicks=0),
        html.Div(id="solver-result", className="text-muted mt-2", style={"fontSize": "0.95rem"}),

        dbc.Button("Save this plan", id="save-plan-btn", color="success", outline=True, className="mt-2", n_clicks=0),
        html.Div(id="save-status", className="text-muted mt-2", style={"fontSize": "0.95rem"}),

        dbc.Card(
            dbc.CardBody([
                html.H6("Insights", className="card-title"),
                html.Div(id="insights")
            ]),
            className="mt-3"
        )
    ]),
    className="shadow-sm"
)


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
                html.H5("Wealth over time (Monte Carlo + optional deterministic)", className="card-title"),
                dcc.Loading(
                    type="circle",
                    children=dcc.Graph(id="paths-graph", config={"displayModeBar": False})
                )
            ]), className="shadow-sm mb-3"),

            dbc.Card(dbc.CardBody([
                html.H5("Distribution of final outcomes", className="card-title"),
                dcc.Loading(
                    type="circle",
                    children=dcc.Graph(id="hist-graph", config={"displayModeBar": False})
                )
            ]), className="shadow-sm"),
        ], width=8),
    ], className="g-3")
])


learn_tab = dbc.Container(fluid=True, className="pt-2", children=[
    dbc.Row([
        dbc.Col([
            dbc.Card(dbc.CardBody([
                html.H4("How to use RiskWise", className="mb-2"),
                html.Ol([
                    html.Li("Pick a Life Goal preset (Emergency Fund / Deposit / Retirement) or choose Custom."),
                    html.Li("Choose a Scenario (Cautious / Balanced / Aggressive)."),
                    html.Li("Optionally enable deterministic forecast to compare ‘average’ vs uncertainty."),
                    html.Li("Optionally enable Compare mode to see two scenarios side-by-side."),
                    html.Li("Optionally enable Real Market Data calibration (SPY/QQQ/IWM)."),
                    html.Li("Set Years, Monthly Contribution, and Goal."),
                    html.Li("Optionally add Fees and Stress Test to see how frictions/shocks change outcomes."),
                    html.Li("Use the Goal Planner to estimate monthly saving needed for a chosen success probability."),
                    html.Li("Use Save this plan to generate a written summary in the Saved Plan tab."),
                ]),
                html.Hr(),
                html.H4("What the charts mean", className="mb-2"),
                html.Ul([
                    html.Li("Median line = typical outcome (50th percentile)."),
                    html.Li("Shaded band = uncertainty range (10th–90th percentiles)."),
                    html.Li("Deterministic line = what a typical ‘single-number’ calculator would show (no volatility)."),
                    html.Li("Histogram = frequency of final outcomes across simulations."),
                    html.Li("Dashed lines = medians and your goal (if set)."),
                ]),
                html.Hr(),
                html.H4("Limitations", className="mb-2"),
                html.Ul([
                    html.Li("Educational tool, not financial advice."),
                    html.Li("Markets can behave differently from historical patterns."),
                    html.Li("Stress tests are simplified educational shocks."),
                ]),
            ]), className="shadow-sm")
        ], width=10),
    ])
])


saved_plan_tab = dbc.Container(fluid=True, className="pt-2", children=[
    dbc.Row([
        dbc.Col([
            dbc.Card(dbc.CardBody([
                html.H4("Saved Plan"),
                html.P(
                    "This tab stores a beginner-friendly snapshot of your current settings and results.",
                    className="text-muted"
                ),
                html.Div(id="saved-plan-body")
            ]), className="shadow-sm")
        ], width=10),
    ], justify="center")
])


onboarding_page = dbc.Container(fluid=True, className="pt-2", children=[
    dbc.Row([
        dbc.Col([
            dbc.Card(dbc.CardBody([
                html.H3("Welcome to RiskWise"),
                html.P(
                    "RiskWise helps you understand uncertainty in saving and investing. "
                    "Instead of one forecast, it shows a range of plausible futures.",
                    className="text-muted"
                ),
                html.H5("How it works (simple)"),
                html.Ol([
                    html.Li("Choose a preset (or Custom)."),
                    html.Li("Pick a risk scenario."),
                    html.Li("Optionally compare with a deterministic ‘average’ forecast."),
                    html.Li("Optionally calibrate using real market data (SPY/QQQ/IWM)."),
                    html.Li("Set years, monthly savings, and goal."),
                    html.Li("Use Fees + Stress Test to see realistic downside."),
                    html.Li("Use Goal Planner to see monthly saving needed for a chosen success probability."),
                    html.Li("Click Save this plan to generate a written explanation in Saved Plan."),
                ]),
                dbc.Alert(
                    "This tool is for education, not financial advice.",
                    color="warning",
                    className="mt-3"
                ),
                dbc.Button("Continue →", id="continue-btn", color="primary", size="lg", className="mt-2")
            ]), className="shadow-sm")
        ], width=10),
    ], justify="center")
])


# -------------------- App setup --------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "RiskWise"
server = app.server

app.layout = dbc.Container(fluid=True, children=[
    navbar,
    dcc.Store(id="onboarded", storage_type="session", data=False),
    dcc.Store(id="saved-plan-store", storage_type="local"),
    html.Div(id="page-body")
])


# -------------------- Callbacks --------------------
@app.callback(Output("page-body", "children"), Input("onboarded", "data"))
def render_page(onboarded):
    if not onboarded:
        return onboarding_page

    return html.Div([
        dbc.Tabs([
            dbc.Tab(label="Dashboard", tab_id="tab-dashboard"),
            dbc.Tab(label="Learn & Method", tab_id="tab-learn"),
            dbc.Tab(label="Saved Plan", tab_id="tab-saved"),
        ], id="tabs", active_tab="tab-dashboard"),
        html.Div(id="tab-content", className="pt-3")
    ])


@app.callback(Output("onboarded", "data"), Input("continue-btn", "n_clicks"), prevent_initial_call=True)
def finish_onboarding(n_clicks):
    return True


@app.callback(Output("tab-content", "children"), Input("tabs", "active_tab"))
def render_tab(active_tab):
    if active_tab == "tab-learn":
        return learn_tab
    if active_tab == "tab-saved":
        return saved_plan_tab
    return dashboard_tab


@app.callback(
    Output("years", "value"),
    Output("monthly", "value"),
    Output("goal", "value"),
    Input("preset", "value")
)
def apply_preset_values(preset):
    return PRESETS.get(preset, PRESETS["custom"])


@app.callback(
    Output("scenario-b-wrap", "style"),
    Output("asset-wrap", "style"),
    Output("crash-wrap", "style"),
    Output("market-missing-note", "style"),
    Input("compare", "value"),
    Input("use_real_data", "value"),
    Input("crash_on", "value"),
)
def toggle_visibility(compare, use_real_data, crash_on):
    compare_on = "on" in (compare or [])
    use_data = "on" in (use_real_data or [])
    crash_enabled = "on" in (crash_on or [])

    scenario_style = {} if compare_on else {"display": "none"}
    crash_style = {} if crash_enabled else {"display": "none"}

    if use_data and MARKET_DATA_AVAILABLE:
        asset_style = {}
        missing_style = {"display": "none"}
    elif use_data and not MARKET_DATA_AVAILABLE:
        asset_style = {"display": "none"}
        missing_style = {}
    else:
        asset_style = {"display": "none"}
        missing_style = {"display": "none"}

    return scenario_style, asset_style, crash_style, missing_style


@app.callback(
    Output("paths-graph", "figure"),
    Output("hist-graph", "figure"),
    Output("insights", "children"),
    Output("data-note", "children"),
    Output("kpi-median", "children"),
    Output("kpi-range", "children"),
    Output("kpi-goalprob", "children"),
    Output("kpi-assumptions", "children"),
    Input("scenario", "value"),
    Input("scenario_b", "value"),
    Input("compare", "value"),
    Input("deterministic_on", "value"),
    Input("years", "value"),
    Input("monthly", "value"),
    Input("goal", "value"),
    Input("use_real_data", "value"),
    Input("asset", "value"),
    Input("fees_pct", "value"),
    Input("crash_on", "value"),
    Input("crash_sev", "value"),
)
def update_graph(scenario, scenario_b, compare, deterministic_on, years, monthly, goal,
                 use_real_data, asset, fees_pct, crash_on, crash_sev):
    compare_on = "on" in (compare or [])
    show_det = "on" in (deterministic_on or [])
    use_data = "on" in (use_real_data or [])

    crash_enabled = "on" in (crash_on or [])
    crash_pct = (float(crash_sev) / 100.0) if crash_enabled else 0.0
    fee_pct = float(fees_pct) / 100.0

    params_a = SCENARIOS[scenario]
    params_b = SCENARIOS[scenario_b]

    data_note = "Using preset scenario assumptions"

    def get_r_vol(default_params):
        nonlocal data_note
        if not use_data or not MARKET_DATA_AVAILABLE:
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

    # Scenario A
    r_a, v_a = get_r_vol(params_a)
    r_a_net = max(-0.50, r_a - fee_pct)

    paths_a = simulate(
        years=years,
        start_balance=1000,
        monthly_contribution=monthly,
        expected_return_annual=r_a_net,
        volatility_annual=v_a,
        inflation_annual=0.02,
        simulations=2000,
        seed=42
    )
    if crash_enabled:
        paths_a = apply_crash(paths_a, crash_pct=crash_pct, seed=1001)

    final_a = paths_a[:, -1]
    summary_a = summarise(final_a)
    p_goal_a = prob_reach_goal(final_a, goal if goal else 0)

    # Scenario B optional
    if compare_on:
        if scenario_b == scenario:
            paths_b, final_b, summary_b = paths_a, final_a, summary_a
            r_b_net = r_a_net
        else:
            r_b, v_b = get_r_vol(params_b)
            r_b_net = max(-0.50, r_b - fee_pct)

            paths_b = simulate(
                years=years,
                start_balance=1000,
                monthly_contribution=monthly,
                expected_return_annual=r_b_net,
                volatility_annual=v_b,
                inflation_annual=0.02,
                simulations=2000,
                seed=43
            )
            if crash_enabled:
                paths_b = apply_crash(paths_b, crash_pct=crash_pct, seed=1002)

            final_b = paths_b[:, -1]
            summary_b = summarise(final_b)
    else:
        paths_b = final_b = summary_b = None

    # ---- Wealth over time ----
    p10_a, p50_a, p90_a = pct_band(paths_a)
    months = np.arange(len(p50_a))

    fig_paths = go.Figure()
    fig_paths.add_trace(go.Scatter(x=months, y=p90_a, mode="lines",
                                   name=f"A {scenario.title()} – 90th", line=dict(width=1)))
    fig_paths.add_trace(go.Scatter(x=months, y=p10_a, mode="lines",
                                   name=f"A {scenario.title()} – 10th (band)", fill="tonexty", line=dict(width=1)))
    fig_paths.add_trace(go.Scatter(x=months, y=p50_a, mode="lines",
                                   name=f"A {scenario.title()} – Median", line=dict(width=3)))

    # NEW: deterministic line for Scenario A
    if show_det:
        det_a = deterministic_path(
            years=years,
            start_balance=1000,
            monthly_contribution=monthly,
            expected_return_annual=r_a_net,
            inflation_annual=0.02
        )
        fig_paths.add_trace(go.Scatter(
            x=np.arange(len(det_a)),
            y=det_a,
            mode="lines",
            name="Deterministic forecast (single line)",
            line=dict(width=3, dash="dash")
        ))

    if compare_on and paths_b is not None and scenario_b != scenario:
        _, p50_b, _ = pct_band(paths_b)
        fig_paths.add_trace(go.Scatter(x=months, y=p50_b, mode="lines",
                                       name=f"B {scenario_b.title()} – Median", line=dict(width=3, dash="dot")))

        # Deterministic for scenario B as well
        if show_det:
            det_b = deterministic_path(
                years=years,
                start_balance=1000,
                monthly_contribution=monthly,
                expected_return_annual=r_b_net,
                inflation_annual=0.02
            )
            fig_paths.add_trace(go.Scatter(
                x=np.arange(len(det_b)),
                y=det_b,
                mode="lines",
                name="Deterministic forecast (B)",
                line=dict(width=2, dash="dashdot")
            ))

    fig_paths.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Months",
        yaxis_title="Balance (real terms)",
        height=360,
        legend_title="Lines",
        hovermode="x unified"
    )

    # ---- Histogram ----
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=final_a, nbinsx=60, name=f"A: {scenario.title()}",
        opacity=0.75, marker=dict(line=dict(width=1))
    ))

    if compare_on and final_b is not None:
        fig_hist.add_trace(go.Histogram(
            x=final_b, nbinsx=60, name=f"B: {scenario_b.title()}",
            opacity=0.55, marker=dict(line=dict(width=1))
        ))
        fig_hist.update_layout(barmode="overlay")

    fig_hist.add_vline(x=summary_a["median"], line_dash="dash",
                       annotation_text="Median (A)", annotation_position="top left")

    if compare_on and summary_b is not None:
        fig_hist.add_vline(x=summary_b["median"], line_dash="dash",
                           annotation_text="Median (B)", annotation_position="top right")

    if goal and goal > 0:
        fig_hist.add_vline(x=goal, line_dash="dash",
                           annotation_text="Goal", annotation_position="top")

    fig_hist.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Final balance (real terms)",
        yaxis_title="Count",
        height=360,
        hovermode="x unified"
    )

    # ---- Insights + KPIs ----
    spread_a = summary_a["p90"] - summary_a["p10"]
    fee_msg = f"Fees applied: {fees_pct:.1f}%/yr"
    crash_msg = f"Stress test ON: crash {crash_sev:.0f}%" if crash_enabled else "Stress test OFF"
    det_msg = "Deterministic comparison ON (single-line calculator forecast shown)" if show_det else "Deterministic comparison OFF"

    if goal and goal > 0:
        if p_goal_a >= 0.75:
            goal_msg = "You’re on track: most simulations reach your goal."
        elif p_goal_a >= 0.40:
            goal_msg = "Your goal is possible, but not guaranteed under these assumptions."
        else:
            goal_msg = "Reaching your goal is unlikely without saving more or extending the time horizon."
    else:
        goal_msg = "Add a goal to see the probability of achieving it."

    insights = html.Ul([
        html.Li(det_msg),
        html.Li(fee_msg),
        html.Li(crash_msg),
        html.Li(f"Most outcomes fall between {money(summary_a['p10'])} and {money(summary_a['p90'])} (spread {money(spread_a)})."),
        html.Li(goal_msg),
    ], className="mb-0")

    kpi_median = money(summary_a["median"])
    kpi_range = f"{money(summary_a['p10'])} – {money(summary_a['p90'])}"
    kpi_goalprob = f"{p_goal_a*100:.1f}%" if goal and goal > 0 else "—"
    kpi_assumptions = f"Net return {r_a_net*100:.1f}% • Vol {v_a*100:.1f}% • Infl 2%"

    return fig_paths, fig_hist, insights, data_note, kpi_median, kpi_range, kpi_goalprob, kpi_assumptions


@app.callback(
    Output("solver-result", "children"),
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
def solve_monthly(n_clicks, scenario, years, goal, target_prob, use_real_data, asset,
                  fees_pct, crash_on, crash_sev):
    if not goal or goal <= 0:
        return "Set a Goal (£) above 0, then click Solve."

    use_data = "on" in (use_real_data or [])
    crash_enabled = "on" in (crash_on or [])
    crash_pct = (float(crash_sev) / 100.0) if crash_enabled else 0.0
    fee_pct = float(fees_pct) / 100.0
    target = float(target_prob) / 100.0

    default_params = SCENARIOS[scenario]

    def get_r_vol():
        if not use_data or not MARKET_DATA_AVAILABLE:
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
    sims_solver = 800
    iters = 12

    for _ in range(iters):
        mid = (lo + hi) / 2.0

        paths = simulate(
            years=years,
            start_balance=1000,
            monthly_contribution=mid,
            expected_return_annual=r_net,
            volatility_annual=v,
            inflation_annual=0.02,
            simulations=sims_solver,
            seed=123
        )
        if crash_enabled:
            paths = apply_crash(paths, crash_pct=crash_pct, seed=2001)

        final_vals = paths[:, -1]
        p = prob_reach_goal(final_vals, goal)

        if p >= target:
            hi = mid
        else:
            lo = mid

    recommended = round(hi / 10) * 10
    return f"To reach {money(goal)} with ~{int(target_prob)}% probability, save about {money(recommended)}/month (under current settings)."


@app.callback(
    Output("saved-plan-store", "data"),
    Output("save-status", "children"),
    Input("save-plan-btn", "n_clicks"),
    State("preset", "value"),
    State("scenario", "value"),
    State("years", "value"),
    State("monthly", "value"),
    State("goal", "value"),
    State("use_real_data", "value"),
    State("asset", "value"),
    State("fees_pct", "value"),
    State("crash_on", "value"),
    State("crash_sev", "value"),
    State("solver-result", "children"),
    prevent_initial_call=True
)
def save_plan(n_clicks, preset, scenario, years, monthly, goal,
              use_real_data, asset, fees_pct, crash_on, crash_sev, solver_text):
    use_data = "on" in (use_real_data or [])
    crash_enabled = "on" in (crash_on or [])
    crash_pct = (float(crash_sev) / 100.0) if crash_enabled else 0.0
    fee_pct = float(fees_pct) / 100.0

    params = SCENARIOS[scenario]
    r, v = params["r"], params["vol"]
    note = "Using preset scenario assumptions"

    if use_data and MARKET_DATA_AVAILABLE:
        try:
            close = fetch_stooq_close_prices(asset)
            mu, vol = estimate_annual_return_vol(close)
            mu = float(np.clip(mu, -0.05, 0.20))
            vol = float(np.clip(vol, 0.05, 0.60))
            r, v = mu, vol
            note = f"Calibrated from {asset.upper()} historical data"
        except Exception:
            note = "Calibration failed (network/data). Using presets."
            r, v = params["r"], params["vol"]

    r_net = max(-0.50, r - fee_pct)

    paths = simulate(
        years=years,
        start_balance=1000,
        monthly_contribution=monthly,
        expected_return_annual=r_net,
        volatility_annual=v,
        inflation_annual=0.02,
        simulations=1600,
        seed=202
    )
    if crash_enabled:
        paths = apply_crash(paths, crash_pct=crash_pct, seed=3001)

    finals = paths[:, -1]
    summ = summarise(finals)
    p_goal = prob_reach_goal(finals, goal if goal else 0)

    snapshot = {
        "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "preset": preset,
        "scenario": scenario,
        "years": years,
        "monthly": monthly,
        "goal": goal,
        "fees_pct": float(fees_pct),
        "crash_enabled": crash_enabled,
        "crash_sev": float(crash_sev) if crash_enabled else 0.0,
        "market_enabled": bool(use_data and MARKET_DATA_AVAILABLE),
        "asset": asset,
        "data_note": note,
        "net_return": float(r_net),
        "volatility": float(v),
        "median": float(summ["median"]),
        "p10": float(summ["p10"]),
        "p90": float(summ["p90"]),
        "goal_prob": float(p_goal),
        "solver_text": str(solver_text) if solver_text else "",
    }

    return snapshot, "Saved. View it in the Saved Plan tab."


@app.callback(Output("saved-plan-body", "children"), Input("saved-plan-store", "data"))
def render_saved_plan(data):
    if not data:
        return dbc.Alert("No saved plan yet. Go to Dashboard → click “Save this plan”.", color="info")

    goal = float(data.get("goal", 0) or 0)
    p_goal = float(data.get("goal_prob", 0) or 0)
    p10 = float(data.get("p10", 0) or 0)
    p90 = float(data.get("p90", 0) or 0)
    med = float(data.get("median", 0) or 0)

    summary_text = plain_english_summary(goal, p_goal, p10, p90)
    actions = recommended_actions(p_goal)

    assumptions = (
        f"Net return: {data['net_return']*100:.1f}% • Volatility: {data['volatility']*100:.1f}% • "
        f"Inflation: 2% • Fees: {data['fees_pct']:.1f}%/yr"
    )

    settings = html.Ul([
        html.Li(f"Preset: {str(data.get('preset','')).title()}"),
        html.Li(f"Scenario: {str(data.get('scenario','')).title()}"),
        html.Li(f"Years: {data.get('years')}"),
        html.Li(f"Monthly contribution: {money(float(data.get('monthly', 0) or 0))}"),
        html.Li(f"Goal: {money(goal) if goal > 0 else 'Not set'}"),
        html.Li(f"Market calibration: {'ON' if data.get('market_enabled') else 'OFF'} ({data.get('data_note','')})"),
        html.Li(f"Stress test: {'ON' if data.get('crash_enabled') else 'OFF'}" +
                (f" ({data.get('crash_sev',0):.0f}% crash)" if data.get('crash_enabled') else "")),
        html.Li(f"Saved at (UTC): {data.get('saved_at','')}"),
    ])

    results = html.Ul([
        html.Li(f"Median outcome: {money(med)}"),
        html.Li(f"10th–90th range: {money(p10)} – {money(p90)}"),
        html.Li(f"Chance of reaching goal: {p_goal*100:.1f}%"),
    ])

    action_list = html.Ul([html.Li(a) for a in actions])

    solver = data.get("solver_text", "")
    solver_block = dbc.Alert(solver, color="success") if solver else dbc.Alert(
        "Tip: Use the Goal Planner on the Dashboard, then save again to include a recommended monthly contribution.",
        color="secondary"
    )

    return html.Div([
        dbc.Alert(summary_text, color="primary"),
        html.H5("Your settings"),
        settings,
        html.Hr(),
        html.H5("Results"),
        results,
        html.P(assumptions, className="text-muted"),
        html.Hr(),
        html.H5("What this means (simple)"),
        html.P(
            "The uncertainty band shows that even with the same average return, outcomes can vary widely due to volatility. "
            "That’s why ‘one-number forecasts’ can be misleading.",
            className="text-muted"
        ),
        html.H5("Recommended actions"),
        action_list,
        html.Hr(),
        html.H5("Goal Planner result"),
        solver_block,
    ])


if __name__ == "__main__":
    app.run(debug=True)
