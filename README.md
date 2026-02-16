🚀 RiskWise - Financial Risk Simulator

Live Demo: https://riskwise-v04j.onrender.com

GitHub: https://github.com/FrankMolloy/Hackonomics-Financial-Risk-Simulator

📌 Overview

Most financial projections show a single number.

Real life doesn’t work like that.

RiskWise is a cloud-deployed financial risk simulator that visualises uncertainty in long-term saving and investing. Instead of projecting a single forecast, it runs thousands of Monte Carlo simulations to model a distribution of possible futures.

The objective is not to predict markets — but to teach probabilistic thinking and improve financial decision-making.

Built for Hackonomics 2026, the project combines:

Economics

Financial literacy

Quantitative modelling

Software engineering

Interactive product design

🎯 Problem

Traditional financial tools often:

Show a single projected value

Ignore volatility and tail risk

Fail to explain uncertainty clearly

Provide limited educational context

This creates false confidence and weak financial planning decisions.

RiskWise addresses this by:

Quantifying uncertainty

Showing downside and upside ranges

Estimating probability of reaching a goal

Translating technical outputs into plain-English insights

🧠 Core Features
📊 Monte Carlo Simulation Engine

2,000–5,000 simulated wealth paths per scenario

Geometric Brownian motion-style modelling

Percentile bands (P10 / P50 / P90)

Distribution-based final outcome analysis

📈 Deterministic vs Monte Carlo Comparison

Single “average” projection shown alongside distribution

Demonstrates why one-line forecasts are misleading

Highlights sequence-of-returns risk

🎯 Goal-Based Planning

Probability of reaching financial targets

Binary search optimisation to compute required monthly contribution

Actionable plan recommendations

📉 Realism Enhancements

Historical market calibration (SPY / QQQ / IWM)

Log-return volatility estimation (annualised, 252-day convention)

Inflation-adjusted results (real purchasing power)

Annual fee drag modelling

Stress-test crash scenarios

Sensitivity analysis (±1% return impact)

🧾 My Plan System

Save scenario snapshots

Generate plain-English plan reports

Export plan as JSON

Compare strategies side-by-side

📚 Educational Modules

Learn & Method tab explaining modelling assumptions

Interactive sequence-of-returns risk demo

Financial literacy insights panel

🔬 Methodology
Return Modelling

Expected returns and volatility can be:

Preset (cautious / balanced / aggressive), or

Calibrated from historical daily log returns (annualised over 252 trading days).

Monte Carlo Engine

Simulates monthly returns using stochastic modelling

Applies contributions and compounding

Adjusts for inflation (real terms)

Computes percentile distributions and goal probability

Goal Solver

Uses binary search optimisation to determine the minimum monthly contribution required to reach a specified probability of success.

Stress Testing

Applies an educational crash shock to illustrate tail risk and downside exposure.

🏗 Architecture
User Input → Simulation Engine → Statistical Analysis → Visualisation Layer → Insight Generator → Plan Storage


Core components:

simulate() – Monte Carlo engine

deterministic_forecast() – average-path comparison

prob_reach_goal() – probability metric

Binary search solver

Dash multi-tab application with state management

Persistent storage via dcc.Store

🛠 Tech Stack

Python

Dash (frontend framework)

Plotly (interactive graphs)

NumPy (numerical computing)

pandas (data analysis)

Requests (market data retrieval)

Gunicorn (WSGI server)

Render (cloud deployment)

Git & GitHub (version control + CI/CD)

☁ Deployment

The application is deployed on Render.

WSGI entrypoint configured with Gunicorn

Continuous deployment enabled

Push to main → automatic rebuild and redeploy

📊 Example Use Cases

Planning a house deposit

Building an emergency fund

Long-term retirement modelling

Comparing cautious vs aggressive strategies

Understanding impact of fees and volatility

⚠ Limitations

Educational tool — not financial advice

Simplified stochastic assumptions

Historical data does not guarantee future performance

Stress test model is illustrative

🧑‍💻 Author

Frank Molloy
Computer Science Student

GitHub: https://github.com/FrankMolloy

LinkedIn: www.linkedin.com/in/frankunderwoodmolloy
