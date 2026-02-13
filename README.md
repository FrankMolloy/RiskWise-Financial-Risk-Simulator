Hackonomics Financial Risk Simulator (RiskWise)

Live Demo: https://riskwise-v04j.onrender.com

RiskWise is a cloud-deployed financial risk simulator designed to help users understand uncertainty in investing and long-term financial planning.

Rather than providing a single forecast, RiskWise runs thousands of Monte Carlo simulations to visualise a realistic range of possible financial outcomes.

Features

Monte Carlo wealth simulation engine (2,000+ simulations per run)

Percentile uncertainty bands (10th–50th–90th percentiles)

Goal probability estimation

Required monthly contribution solver (binary search optimisation)

Real-market calibration using historical SPY / QQQ / IWM data

Annual fee drag modelling

Stress-test crash simulation mode

Inflation-adjusted results (real terms)

Interactive dashboard with onboarding tutorial

Methodology

Expected returns and volatility can be:

Preset (cautious / balanced / aggressive scenarios), or

Calibrated using historical daily log returns annualised over 252 trading days.

Monte Carlo simulations generate wealth paths assuming geometric Brownian motion-style return modelling.

Final outcomes are analysed using percentile statistics to quantify uncertainty and tail risk.

The goal planner uses binary search to compute the minimum monthly contribution required to achieve a specified probability of success.

Tech Stack

Python

Dash

Plotly

NumPy

pandas

Requests

Gunicorn (WSGI server)

Render (cloud deployment)

Git & GitHub (CI/CD workflow)

Deployment

The application is deployed via Render and configured with a WSGI entrypoint using Gunicorn.

Continuous deployment is enabled: pushing to the main branch automatically rebuilds and redeploys the application.

Limitations

Educational tool only — not financial advice.

Uses simplified stochastic modelling assumptions.

Historical data may not reflect future performance.

Author

Frank Molloy
Computer Science Student
GitHub: https://github.com/FrankMolloy
