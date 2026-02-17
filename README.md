🚀 RiskWise – Monte Carlo Financial Risk Simulator

Live Demo: https://riskwise-v04j.onrender.com
Built for Hackonomics 2026

What It Does

RiskWise is a cloud-deployed financial risk simulator that models long-term investment outcomes under uncertainty.

Instead of showing a single forecast, it runs 2,000–5,000 Monte Carlo simulations per scenario to visualise a distribution of possible outcomes.

It helps users:

Understand downside and upside risk

Estimate probability of reaching financial goals

Compare strategies under volatility

Why It Matters

Traditional financial tools show one projected number.

RiskWise demonstrates why that is misleading — by modelling volatility, tail risk, and sequence-of-returns effects using stochastic simulation.

The aim is educational: to promote probabilistic thinking in financial decision-making.

Core Features

Monte Carlo simulation engine (2,000–5,000 runs)

Percentile bands (P10 / P50 / P90)

Goal probability modelling

Binary search solver for required monthly contribution

Stress testing (crash scenarios + fee drag)

Interactive Dash + Plotly dashboard

Save & compare financial plans

Methodology (Brief Technical Summary)

Simulates monthly returns using stochastic modelling

Optional calibration from historical ETF log returns (annualised, 252-day convention)

Inflation-adjusted (real terms) projections

Percentile distribution analysis

Tech Stack

Python, NumPy, pandas
Dash + Plotly
Gunicorn (WSGI)
Render (cloud deployment)
GitHub CI/CD

Architecture

User Input → Simulation Engine → Statistical Analysis → Visualisation → Insight Generation → Plan Storage

Limitations

Educational tool — not financial advice.
Simplified stochastic assumptions.

Author

Frank Molloy
GitHub: […](https://github.com/FrankMolloy)
LinkedIn: http://www.linkedin.com/in/frankunderwoodmolloy
