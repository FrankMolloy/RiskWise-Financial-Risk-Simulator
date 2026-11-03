# Local development and deployment

## Setup

Use Python 3.12; the pinned NumPy/pandas versions require a recent Python. Clone the existing project:

```bash
git clone https://github.com/FrankMolloy/RiskWise-Financial-Risk-Simulator.git
cd RiskWise-Financial-Risk-Simulator
python -m venv .venv
```

Activate in PowerShell with `.\.venv\Scripts\Activate.ps1`, or macOS/Linux with `source .venv/bin/activate`. If PowerShell blocks activation, call `.\.venv\Scripts\python.exe` directly instead of changing machine policy.

```bash
python -m pip install -r requirements.txt
python -m pip check
python SRC/app.py
```

Visit `http://127.0.0.1:8050`. Debug mode is off by default. No account, database, credentials or environment file is needed. Preset simulations work without market-data access; the Bootstrap theme still requires its CDN. Optional historical calibration requires outbound HTTPS to Stooq. The CLI remains available via `python SRC/main.py` and opens Matplotlib charts.

## Verification

Set the source import path once per terminal:

```powershell
# Windows PowerShell
$env:PYTHONPATH="SRC"
python -m unittest discover -s tests -v
```

```bash
# macOS/Linux
PYTHONPATH=SRC python -m unittest discover -s tests -v
```

Tests cover closed-form balances, contribution timing, inflation, seeded regression, RNG isolation, terminal mean, invalid inputs, empirical percentiles, calibration parsing/cache/failure, solver bounds, crash behavior and Dash routes/callbacks. Market-data unit tests mock HTTP and run offline.

The final review ran 42 tests in both the development environment and a second clean environment containing only the pinned application dependencies. Its additional checks and remaining gaps are recorded in [FINAL_REVIEW.md](FINAL_REVIEW.md).

```bash
python -m compileall -q SRC
python -m pip check
# Optional audit tool; not an application dependency
python -m pip install pip-audit==2.10.1
python -m pip_audit -r requirements.txt --no-deps --disable-pip
```

There is no frontend bundling/build step: Dash serves the application directly. Tests do not measure production throughput or certify statistical realism. The Dash callback protocol is not a separately designed REST API.

## Production

The existing `Procfile` is unchanged:

```text
web: cd SRC && gunicorn app:server
```

Gunicorn is a Unix production server and cannot run natively on Windows. Windows verification uses Flask's test client and the Dash development server; the added Ubuntu CI job starts the actual Gunicorn entrypoint and requests `/_dash-layout`.

For a Render web service, use Python 3.12, build with `python -m pip install -r requirements.txt` and use the existing Procfile command as the start command. If configuring a port explicitly in Render's Linux shell, use `cd SRC && gunicorn app:server --bind 0.0.0.0:$PORT`. The audit did not access or alter Render's private settings, build hooks, environment or auto-deploy configuration.

The GitHub workflow runs tests and a dependency audit on pushes/PRs. It **does not deploy**. Neither its first hosted run nor the Linux smoke test can be verified until changes are pushed; no push was made during this audit.

## Storage and operational limits

Plans live in browser `localStorage`, without authentication or server persistence. JSON exports contain selected settings and saved solver text, not simulated paths or a complete reproducible market-data snapshot. Clearing browser data loses plans. The cache is process-local. Callbacks execute synchronously and allocate full path arrays; there is no job queue, rate limiter, server-enforced request budget or load-test evidence. Slider limits are UI controls. Evaluate those boundaries before exposing this as a general public computation API.
