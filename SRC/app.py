from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np

from simulator import simulate
from metrics import summarise, prob_reach_goal
from market_data import fetch_stooq_close_prices, estimate_annual_return_vol

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

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True
)
app.title = "RiskWise"

server = app.server

def money(x: float) -> str:
    return f"£{x:,.0f}"


def pct_band(paths: np.ndarray):
    p10 = np.percentile(paths, 10, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p90 = np.percentile(paths, 90, axis=0)
    return p10, p50, p90


def apply_crash(paths: np.ndarray, crash_pct: float, seed: int = 999) -> np.ndarray:
    """
    Simple stress-test: for each simulated path, pick a random crash month and
    apply a one-off drawdown of crash_pct to all future balances.
    Educational approximation, but very effective for illustrating downside risk.
    """
    if crash_pct <= 0:
        return paths

    out = paths.copy()
    rng = np.random.default_rng(seed)
    n_sims, n_steps = out.shape

    # avoid crashing at month 0 / too late
    lo = min(12, n_steps - 2)
    hi = max(lo + 1, n_steps - 12)
    crash_months = rng.integers(low=lo, high=hi, size=n_sims)

    factor = 1.0 - crash_pct
    for i in range(n_sims):
        m = int(crash_months[i])
        out[i, m:] *= factor

    return out


# -------------------- Layout Components --------------------

def kpi_card(title: str, value_id: str):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="text-muted"),
            html.H3(id=value_id, className="mb-0")
        ]),
        className="shadow-sm"
    )


# Centered navbar brand
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

        html.Label("Scenario B (Comparison)", className="mt-2"),
        dcc.Dropdown(
            options=[{"label": k.title(), "value": k} for k in SCENARIOS.keys()],
            value="aggressive",
            id="scenario_b",
            clearable=False
        ),

        html.Hr(),

        dbc.Checklist(
            options=[{"label": "Use real market data (calibrate return/vol)", "value": "on"}],
            value=[],
            id="use_real_data",
            switch=True,
            className="mt-2"
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
            clearable=False
        ),

        html.Div(id="data-note", className="text-muted mt-2", style={"fontSize": "0.9rem"}),

        html.Hr(),

        # 1) FEES (annual)
        html.Label("Annual fees (%)", className="mt-2"),
        dcc.Slider(
            0.0, 2.0, 0.1,
            value=0.3,
            id="fees_pct",
            marks={0.0: "0%", 0.5: "0.5%", 1.0: "1%", 1.5: "1.5%", 2.0: "2%"}
        ),

        # 3) STRESS TEST TOGGLE
        dbc.Checklist(
            options=[{"label": "Stress test: include a market crash", "value": "on"}],
            value=[],
            id="crash_on",
            switch=True,
            className="mt-3"
        ),

        html.Label("Crash severity (%)", className="mt-2"),
        dcc.Slider(
            10, 60, 5,
            value=30,
            id="crash_sev",
            marks={10: "10%", 30: "30%", 50: "50%", 60: "60%"}
        ),

        html.Hr(),

        html.Label("Years", className="mt-2"),
        dcc.Slider(1, 40, 1, value=10, id="years",
                   marks={1: "1", 10: "10", 20: "20", 30: "30", 40: "40"}),

        html.Label("Monthly Contribution", className="mt-3"),
        dcc.Slider(0, 2000, 50, value=200, id="monthly",
                   marks={0: "0", 500: "500", 1000: "1000", 1500: "1500", 2000: "2000"}),

        html.Label("Goal (£)", className="mt-3"),
        dbc.Input(type="number", value=50000, id="goal", min=0, step=1000),

        html.Hr(),

        # 2) SOLVER
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
                html.H5("Wealth over time (Median + Uncertainty Band)", className="card-title"),
                dcc.Graph(id="paths-graph", config={"displayModeBar": False})
            ]), className="shadow-sm mb-3"),

            dbc.Card(dbc.CardBody([
                html.H5("Distribution of final outcomes", className="card-title"),
                dcc.Graph(id="hist-graph", config={"displayModeBar": False})
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
                    html.Li("Optionally enable Compare mode to see two scenarios side-by-side."),
                    html.Li("Optionally enable Real Market Data calibration (SPY/QQQ/IWM)."),
                    html.Li("Set Years, Monthly Contribution, and Goal."),
                    html.Li("Optionally add Fees and Stress Test to see how realistic frictions/shocks change outcomes."),
                    html.Li("Use the Goal Planner to estimate the monthly saving needed for a chosen success probability."),
                ]),
                html.Hr(),
                html.H4("What the charts mean", className="mb-2"),
                html.Ul([
                    html.Li("Median line = typical outcome (50th percentile)."),
                    html.Li("Shaded band = uncertainty range (10th–90th percentiles)."),
                    html.Li("Histogram = frequency of final outcomes across simulations."),
                    html.Li("Dashed lines = median(s) and your goal (if set)."),
                ]),
                html.Hr(),
                html.H4("Method (simple)", className="mb-2"),
                html.Ul([
                    html.Li("RiskWise generates thousands of simulated return paths (Monte Carlo)."),
                    html.Li("If calibration is enabled, it estimates return & volatility from historical price data (log returns, annualised)."),
                    html.Li("Fees reduce expected return each year (a common real-world drag)."),
                    html.Li("Stress test applies an educational crash shock to highlight tail risk."),
                    html.Li("Outputs are inflation-adjusted so values are in 'today’s money'."),
                ]),
                html.Hr(),
                html.H4("Limitations", className="mb-2"),
                html.Ul([
                    html.Li("Educational tool, not financial advice."),
                    html.Li("Markets can crash and behave differently from historical patterns."),
                    html.Li("The crash stress test is a simplified model to illustrate downside risk."),
                ]),
            ]), className="shadow-sm")
        ], width=10),
    ])
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
                    html.Li("Optionally calibrate using real market data (SPY/QQQ/IWM)."),
                    html.Li("Set years, monthly savings, and goal."),
                    html.Li("Use Fees + Stress Test to see realistic downside."),
                    html.Li("Use Goal Planner to see monthly saving needed for a chosen success probability."),
                ]),
                html.Hr(),
                html.H5("What the charts mean"),
                html.Ul([
                    html.Li("Median = typical outcome."),
                    html.Li("Band = uncertainty range."),
                    html.Li("Histogram = how often outcomes occur."),
                    html.Li("Dashed lines = medians and goal."),
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


# -------------------- App Layout --------------------

app.layout = dbc.Container(fluid=True, children=[
    navbar,
    dcc.Store(id="onboarded", storage_type="session", data=False),
    html.Div(id="page-body")
])


# -------------------- Callbacks --------------------

@app.callback(
    Output("page-body", "children"),
    Input("onboarded", "data")
)
def render_page(onboarded):
    if not onboarded:
        return onboarding_page

    return html.Div([
        dbc.Tabs([
            dbc.Tab(label="Dashboard", tab_id="tab-dashboard"),
            dbc.Tab(label="Learn & Method", tab_id="tab-learn"),
        ], id="tabs", active_tab="tab-dashboard"),
        html.Div(id="tab-content", className="pt-3")
    ])


@app.callback(
    Output("onboarded", "data"),
    Input("continue-btn", "n_clicks"),
    prevent_initial_call=True
)
def finish_onboarding(n_clicks):
    return True


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def render_tab(active_tab):
    return learn_tab if active_tab == "tab-learn" else dashboard_tab


@app.callback(
    Output("years", "value"),
    Output("monthly", "value"),
    Output("goal", "value"),
    Input("preset", "value")
)
def apply_preset(preset):
    return PRESETS.get(preset, PRESETS["custom"])


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
    Input("years", "value"),
    Input("monthly", "value"),
    Input("goal", "value"),
    Input("use_real_data", "value"),
    Input("asset", "value"),
    Input("fees_pct", "value"),
    Input("crash_on", "value"),
    Input("crash_sev", "value"),
)
def update_graph(scenario, scenario_b, compare, years, monthly, goal,
                 use_real_data, asset, fees_pct, crash_on, crash_sev):
    compare_on = "on" in (compare or [])
    use_data = "on" in (use_real_data or [])

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

    # Apply fees (simple & realistic): net expected return
    r_a, v_a = get_r_vol(params_a)
    r_a_net = max(-0.50, r_a - fee_pct)  # guardrail

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

    if compare_on:
        if scenario_b == scenario:
            paths_b, final_b, summary_b, p_goal_b = paths_a, final_a, summary_a, p_goal_a
            r_b_net, v_b = r_a_net, v_a
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
            p_goal_b = prob_reach_goal(final_b, goal if goal else 0)
    else:
        paths_b = final_b = summary_b = p_goal_b = None
        r_b_net = v_b = None

    # ---- Wealth over time: Median + Uncertainty Band ----
    p10_a, p50_a, p90_a = pct_band(paths_a)
    months = np.arange(len(p50_a))

    fig_paths = go.Figure()

    fig_paths.add_trace(go.Scatter(
        x=months, y=p90_a,
        mode="lines",
        name=f"A {scenario.title()} – 90th percentile",
        line=dict(width=1),
        hovertemplate="Month %{x}<br>P90: %{y:.0f}<extra></extra>"
    ))
    fig_paths.add_trace(go.Scatter(
        x=months, y=p10_a,
        mode="lines",
        name=f"A {scenario.title()} – 10th percentile (band)",
        fill="tonexty",
        line=dict(width=1),
        hovertemplate="Month %{x}<br>P10: %{y:.0f}<extra></extra>"
    ))
    fig_paths.add_trace(go.Scatter(
        x=months, y=p50_a,
        mode="lines",
        name=f"A {scenario.title()} – Median",
        line=dict(width=3),
        hovertemplate="Month %{x}<br>Median: %{y:.0f}<extra></extra>"
    ))

    if compare_on and paths_b is not None and scenario_b != scenario:
        p10_b, p50_b, p90_b = pct_band(paths_b)

        fig_paths.add_trace(go.Scatter(
            x=months, y=p90_b,
            mode="lines",
            name=f"B {scenario_b.title()} – 90th percentile",
            line=dict(width=1, dash="dot"),
            hovertemplate="Month %{x}<br>P90: %{y:.0f}<extra></extra>"
        ))
        fig_paths.add_trace(go.Scatter(
            x=months, y=p10_b,
            mode="lines",
            name=f"B {scenario_b.title()} – 10th percentile (band)",
            fill="tonexty",
            line=dict(width=1, dash="dot"),
            hovertemplate="Month %{x}<br>P10: %{y:.0f}<extra></extra>"
        ))
        fig_paths.add_trace(go.Scatter(
            x=months, y=p50_b,
            mode="lines",
            name=f"B {scenario_b.title()} – Median",
            line=dict(width=3, dash="dot"),
            hovertemplate="Month %{x}<br>Median: %{y:.0f}<extra></extra>"
        ))

    fig_paths.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Months",
        yaxis_title="Balance (real terms)",
        height=360,
        legend_title="Lines",
    )

    # ---- Histogram: clearer bins + labelled lines ----
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=final_a,
        nbinsx=60,
        name=f"A: {scenario.title()}",
        opacity=0.75,
        marker=dict(line=dict(width=1))
    ))

    if compare_on and final_b is not None:
        fig_hist.add_trace(go.Histogram(
            x=final_b,
            nbinsx=60,
            name=f"B: {scenario_b.title()}",
            opacity=0.55,
            marker=dict(line=dict(width=1))
        ))
        fig_hist.update_layout(barmode="overlay")

    fig_hist.add_vline(
        x=summary_a["median"],
        line_dash="dash",
        annotation_text="Median (A)",
        annotation_position="top left"
    )
    if compare_on and summary_b is not None:
        fig_hist.add_vline(
            x=summary_b["median"],
            line_dash="dash",
            annotation_text="Median (B)",
            annotation_position="top right"
        )
    if goal and goal > 0:
        fig_hist.add_vline(
            x=goal,
            line_dash="dash",
            annotation_text="Goal",
            annotation_position="top"
        )

    fig_hist.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Final balance (real terms)",
        yaxis_title="Count",
        height=360
    )

    # ---- Insights + KPIs ----
    spread_a = summary_a["p90"] - summary_a["p10"]

    fee_msg = f"Fees applied: {fees_pct:.1f}%/yr"
    crash_msg = f"Stress test ON: crash {crash_sev:.0f}%" if crash_enabled else "Stress test OFF"

    if goal and goal > 0:
        if p_goal_a >= 0.75:
            goal_msg = "You’re on track: most simulations reach your goal."
        elif p_goal_a >= 0.40:
            goal_msg = "Your goal is possible, but not guaranteed under these assumptions."
        else:
            goal_msg = "Reaching your goal is unlikely without saving more or extending the time horizon."
    else:
        goal_msg = "Add a goal to see the probability of achieving it."

    if compare_on and summary_b is not None:
        spread_b = summary_b["p90"] - summary_b["p10"]
        insights = html.Ul([
            html.Li(fee_msg),
            html.Li(crash_msg),
            html.Li(f"A ({scenario.title()}): {money(summary_a['p10'])}–{money(summary_a['p90'])} (spread {money(spread_a)})."),
            html.Li(f"B ({scenario_b.title()}): {money(summary_b['p10'])}–{money(summary_b['p90'])} (spread {money(spread_b)})."),
            html.Li(goal_msg),
        ], className="mb-0")
    else:
        insights = html.Ul([
            html.Li(fee_msg),
            html.Li(crash_msg),
            html.Li(f"Most outcomes fall between {money(summary_a['p10'])} and {money(summary_a['p90'])} (spread {money(spread_a)})."),
            html.Li(goal_msg),
        ], className="mb-0")

    kpi_median = money(summary_a["median"])
    kpi_range = f"{money(summary_a['p10'])} – {money(summary_a['p90'])}"
    kpi_goalprob = f"{p_goal_a*100:.1f}%" if goal and goal > 0 else "—"
    kpi_assumptions = f"Net return {r_a_net*100:.1f}% • Vol {v_a*100:.1f}% • Infl 2%"

    return (
        fig_paths, fig_hist,
        insights, data_note,
        kpi_median, kpi_range, kpi_goalprob, kpi_assumptions
    )


# 2) Goal Planner solver (binary search)
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

    # Binary search monthly contribution
    lo, hi = 0.0, 5000.0
    sims_solver = 800  # keep fast
    iters = 12         # ~ precision enough for UI

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

    recommended = round(hi / 10) * 10  # neat rounding
    return f"To reach {money(goal)} with ~{int(target_prob)}% probability, save about {money(recommended)}/month (under current settings)."


if __name__ == "__main__":
    app.run(debug=True)
