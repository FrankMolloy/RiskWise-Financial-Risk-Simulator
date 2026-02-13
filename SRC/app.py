from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np

from simulator import simulate
from metrics import summarise, prob_reach_goal
from market_data import fetch_stooq_close_prices, estimate_annual_return_vol


# Scenario presets (used when real-data toggle is off, or if calibration fails)
SCENARIOS = {
    "cautious":   {"r": 0.05, "vol": 0.10},
    "balanced":   {"r": 0.07, "vol": 0.15},
    "aggressive": {"r": 0.09, "vol": 0.20},
}

# Life-goal presets (years, monthly, goal)
PRESETS = {
    "emergency": (2, 300, 3000),
    "deposit": (5, 500, 20000),
    "retirement": (30, 400, 500000),
    "custom": (10, 200, 50000),
}


def money(x: float) -> str:
    return f"£{x:,.0f}"


app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "RiskWise"


app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col([
            html.H2("RiskWise", className="mt-3"),
            html.P(
                "A Monte Carlo financial simulator that visualises uncertainty, inflation effects, and goal probability.",
                className="text-muted"
            ),
        ], width=8),
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Inputs", className="card-title"),

                    # Life Goal Preset dropdown
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

                    # Real data calibration
                    dbc.Checklist(
                        options=[{"label": "Use real market data (calibrate return/vol)", "value": "on"}],
                        value=[],
                        id="use_real_data",
                        switch=True,
                        className="mt-3"
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

                    html.Label("Years", className="mt-3"),
                    dcc.Slider(
                        1, 40, 1,
                        value=10,
                        id="years",
                        marks={1: "1", 10: "10", 20: "20", 30: "30", 40: "40"}
                    ),

                    html.Label("Monthly Contribution", className="mt-3"),
                    dcc.Slider(
                        0, 2000, 50,
                        value=200,
                        id="monthly",
                        marks={0: "0", 500: "500", 1000: "1000", 1500: "1500", 2000: "2000"}
                    ),

                    html.Label("Goal (£)", className="mt-3"),
                    dbc.Input(type="number", value=50000, id="goal", min=0, step=1000),

                    html.Hr(),

                    html.Div(id="stats-cards"),

                    dbc.Card(
                        dbc.CardBody([
                            html.H6("Insights", className="card-title"),
                            html.Div(id="insights")
                        ]),
                        className="mt-3"
                    )
                ])
            ], className="shadow-sm")
        ], width=4),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Simulated Paths", className="card-title"),
                    dcc.Graph(id="paths-graph", config={"displayModeBar": False})
                ])
            ], className="shadow-sm mb-3"),

            dbc.Card([
                dbc.CardBody([
                    html.H5("Outcome Distribution", className="card-title"),
                    dcc.Graph(id="hist-graph", config={"displayModeBar": False})
                ])
            ], className="shadow-sm")
        ], width=8),
    ], className="mt-3 mb-4")
])


# Apply preset to years/monthly/goal
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
    Output("stats-cards", "children"),
    Output("insights", "children"),
    Output("data-note", "children"),
    Input("scenario", "value"),
    Input("scenario_b", "value"),
    Input("compare", "value"),
    Input("years", "value"),
    Input("monthly", "value"),
    Input("goal", "value"),
    Input("use_real_data", "value"),
    Input("asset", "value"),
)
def update_graph(scenario, scenario_b, compare, years, monthly, goal, use_real_data, asset):
    compare_on = "on" in (compare or [])
    use_data = "on" in (use_real_data or [])

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

            # Guardrails so weird estimates don't break demo:
            mu = float(np.clip(mu, -0.05, 0.20))   # -5% to +20% expected return
            vol = float(np.clip(vol, 0.05, 0.60))  # 5% to 60% volatility

            data_note = f"Calibrated from {asset.upper()} historical data"
            return mu, vol
        except Exception:
            data_note = "Calibration failed (network/data). Using presets."
            return default_params["r"], default_params["vol"]

    # Get parameters for A (and B if compare on)
    r_a, v_a = get_r_vol(params_a)

    # Scenario A simulation
    paths_a = simulate(
        years=years,
        start_balance=1000,
        monthly_contribution=monthly,
        expected_return_annual=r_a,
        volatility_annual=v_a,
        inflation_annual=0.02,
        simulations=2000,
        seed=42
    )
    final_a = paths_a[:, -1]
    summary_a = summarise(final_a)
    p_goal_a = prob_reach_goal(final_a, goal if goal else 0)

    # Scenario B simulation (only if compare enabled)
    if compare_on:
        # If scenarios are identical, reuse A so users aren’t confused by Monte Carlo noise
        if scenario_b == scenario:
            paths_b = paths_a
            final_b = final_a
            summary_b = summary_a
            p_goal_b = p_goal_a
            r_b, v_b = r_a, v_a
        else:
            r_b, v_b = get_r_vol(params_b)
            paths_b = simulate(
                years=years,
                start_balance=1000,
                monthly_contribution=monthly,
                expected_return_annual=r_b,
                volatility_annual=v_b,
                inflation_annual=0.02,
                simulations=2000,
                seed=43
            )
            final_b = paths_b[:, -1]
            summary_b = summarise(final_b)
            p_goal_b = prob_reach_goal(final_b, goal if goal else 0)
    else:
        paths_b = None
        final_b = None
        summary_b = None
        p_goal_b = None
        r_b, v_b = None, None

    # ----- Paths plot -----
    fig_paths = go.Figure()
    sample_n = min(40, paths_a.shape[0])
    for i in range(sample_n):
        fig_paths.add_trace(go.Scatter(
            y=paths_a[i],
            mode="lines",
            showlegend=False,
            opacity=0.35
        ))

    if compare_on and paths_b is not None and scenario_b != scenario:
        sample_n_b = min(40, paths_b.shape[0])
        for i in range(sample_n_b):
            fig_paths.add_trace(go.Scatter(
                y=paths_b[i],
                mode="lines",
                showlegend=False,
                opacity=0.25
            ))

    fig_paths.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Months",
        yaxis_title="Balance (real terms)",
        height=320,
        title="Simulated Wealth Paths"
    )

    # ----- Histogram -----
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=final_a,
        nbinsx=60,
        name=f"A: {scenario.title()}",
        opacity=0.75
    ))

    if compare_on and final_b is not None:
        fig_hist.add_trace(go.Histogram(
            x=final_b,
            nbinsx=60,
            name=f"B: {scenario_b.title()}",
            opacity=0.55
        ))
        fig_hist.update_layout(barmode="overlay")

    # Median lines for A/B + Goal line
    fig_hist.add_vline(x=summary_a["median"], line_dash="dash")
    if compare_on and summary_b is not None:
        fig_hist.add_vline(x=summary_b["median"], line_dash="dash")
    if goal and goal > 0:
        fig_hist.add_vline(x=goal, line_dash="dash")

    fig_hist.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Final balance (real terms)",
        yaxis_title="Count",
        height=320,
        title="Distribution of Final Outcomes"
    )

    # ----- Stats cards -----
    if not compare_on:
        stats_cards = dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Median", className="text-muted"),
                html.H4(money(summary_a["median"]))
            ]), className="mb-2"), width=6),

            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("10th–90th percentile", className="text-muted"),
                html.H4(f"{money(summary_a['p10'])} – {money(summary_a['p90'])}")
            ]), className="mb-2"), width=6),

            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Goal probability", className="text-muted"),
                html.H4(f"{p_goal_a*100:.1f}%")
            ]), className="mb-2"), width=6),

            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Assumptions used", className="text-muted"),
                html.Small(f"Return {r_a*100:.1f}%, Vol {v_a*100:.1f}%, Inflation 2%")
            ]), className="mb-2"), width=6),
        ])
    else:
        stats_cards = dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div(f"Median (A: {scenario.title()})", className="text-muted"),
                html.H4(money(summary_a["median"]))
            ]), className="mb-2"), width=6),

            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div(f"Median (B: {scenario_b.title()})", className="text-muted"),
                html.H4(money(summary_b["median"]))
            ]), className="mb-2"), width=6),

            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Goal probability (A)", className="text-muted"),
                html.H4(f"{p_goal_a*100:.1f}%")
            ]), className="mb-2"), width=6),

            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Goal probability (B)", className="text-muted"),
                html.H4(f"{p_goal_b*100:.1f}%")
            ]), className="mb-2"), width=6),
        ])

    # ----- Insights -----
    spread_a = summary_a["p90"] - summary_a["p10"]

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
            html.Li(f"A ({scenario.title()}): typical outcomes {money(summary_a['p10'])} to {money(summary_a['p90'])} (spread {money(spread_a)})."),
            html.Li(f"B ({scenario_b.title()}): typical outcomes {money(summary_b['p10'])} to {money(summary_b['p90'])} (spread {money(spread_b)})."),
            html.Li("Higher volatility usually increases uncertainty (wider spread), even if the median rises."),
            html.Li(goal_msg),
        ], className="mb-0")
    else:
        insights = html.Ul([
            html.Li(f"Most outcomes fall between {money(summary_a['p10'])} and {money(summary_a['p90'])} (spread {money(spread_a)})."),
            html.Li(goal_msg),
            html.Li("Try Compare mode to see how risk changes the spread of outcomes."),
        ], className="mb-0")

    return fig_paths, fig_hist, stats_cards, insights, data_note


if __name__ == "__main__":
    app.run(debug=True)
