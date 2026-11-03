# Final engineering review

Reviewed 3 November 2026 against original commit `2cb24bc`. No commit, push or deployment was made.

**Recommendation: GO to publish the source changes as an educational portfolio project. This is not a production deployment sign-off.** Hosted CI, actual Linux/Gunicorn execution and the updated Render deployment remain unverified. Check Render auto-deploy settings before any future push if publishing must not also deploy.

## Diff review and fixes

Reviewed every tracked modification, deletion and new source/test/document/workflow file. The changes remain focused on correctness, testability, documentation and dependency fixes. No runtime packages were added or removed: exactly eight of the original 37 pins changed. The requirements file changed from UTF-16 to UTF-8, explaining Git's binary diff. The three deleted files are generated CPython bytecode. The existing Procfile and public WSGI object `app:server` are unchanged.

The final review fixed a one-month helper boundary: a crash could previously haircut the initial balance when passed a two-column path. Month zero is now preserved. This does not change any whole-year application output. It also corrected the dashboard insight label so failed calibration is described as preset fallback, and ignored Ruff's generated cache.

## Independent numerical comparison

Executed the original simulation source directly from Git, alongside the current implementation in the same pinned NumPy environment. Compared complete arrays across 72 combinations: horizons 0/1/10/40 years, 1/1,000 paths, seeds 0/42/43 and cautious/balanced-net/aggressive return-volatility assumptions, with £1,000 initial capital and £200/month.

**All 5,549,544 balance values were bit-for-bit equal.** Positive-horizon crash outputs and deterministic overlays were also exactly equal to their original functions. This establishes regression compatibility for the tested grid, not numerical equivalence on every platform or every possible input.

The recurrence remains `B_t = B_(t-1) * (1 + R_t) + C`, with independent `R_t ~ Normal(mu/12, sigma/sqrt(12))` and real balance `B_t / (1+inflation)^(t/12)`. P10/P50/P90 remain empirical, linearly interpolated per-month quantiles. They are not confidence intervals, worst-case bounds or single sample paths. Goal probability counts final balances **at least** the real-terms target.

Intentional user-visible differences:

- The default solver now recommends £460/month rather than £450: rounding upward avoids returning a contribution below its sample target. Infeasible goals now produce a message rather than a misleading £5,000 recommendation. Already-funded goals can return zero.
- After Solve, changing relevant settings recalculates the recommendation. Initial rendering waits for a click. The solver still uses 900 fixed-seed paths, rather than the dashboard's sample.
- Invalid/non-finite simulation, metric and calibration inputs are rejected. Negative contributions or initial balances that the old module accepted now raise `ValueError`; withdrawals are outside the app's documented scope.
- RNG state is local rather than process-global. Fixed integer seeds retain original draws; code depending on simulation calls to reseed unrelated NumPy calculations will intentionally behave differently.
- Zero-horizon stress preserves initial capital; one-month helper stress can only affect month one. Normal application horizons are unchanged. Local debug mode is off; UTC timestamp text format is preserved.

## Tests and checks

| Check | Final result |
| --- | --- |
| Existing 34 tests | Passed before additions |
| Expanded suite, development environment | 42 passed |
| Expanded suite, clean Python 3.12.14 environment with application pins only | 42 passed |
| Compilation and `pip check` in clean environment | Passed; no broken requirements |
| Ruff `F,E9` on all changed source modules and tests | Passed |
| Ruff across all source | Eight pre-existing F401 unused-import findings in `SRC/init.py` and `SRC/main.py`; retained to avoid unrelated cleanup |
| Bandit across all Python source | Zero findings; zero scan errors |
| `pip-audit` on the 37 application pins | No known vulnerabilities at review time |
| Candidate-file and 19-revision credential-pattern scans | No matches; no candidate environment, private-key or certificate files |
| CLI/headless Matplotlib PNG rendering | Completed; Pillow integration passed |
| Local headless Edge | Charts, solver recalculation/infeasibility, plan save/download, sequence and method tabs passed; no JavaScript or local HTTP 5xx errors |
| README Mermaid | Parsed and rendered successfully with Mermaid 11; eight nodes, reflecting real components |
| Documentation links/screenshots | Local Markdown targets resolve; dashboard and sequence screenshots refreshed from the clean-environment app and visually inspected |
| Linux dependency resolution | All 37 pins downloaded as Python 3.12/Linux x86-64-compatible wheels; no Linux execution implied |

The original tests include independent annuity and terminal-mean oracles, inclusive goal-boundary checks and failure behavior, alongside regression and wiring tests. The final eight additions cover hand-calculated interpolated quantiles, non-finite goals, analytic/stochastic solver feasibility, invalid search configuration, one-month crashes, insufficient history and cache expiry. Tests that count graph traces are smoke checks, not mathematical validation.

Measured **branch-aware coverage** from the unit suite: simulator, metrics, planning and market-data modules 100%; projections 96%; app 61%; all SRC files 66%, including the legacy CLI/diagnostics. Browser and CLI smoke runs are separate and are not included in these percentages. Coverage does not establish model realism, and UI branches remain incompletely covered.

## Eight dependency updates

All eight installed metadata records accept Python 3.12. The second clean install and Linux wheel resolution succeeded. Release-note review did not identify an API used by this project that requires migration, but these upgrades are not universally behavior-neutral.

| Dependency | Change | Compatibility assessment |
| --- | --- | --- |
| certifi | 2026.1.4 → 2026.10.18 | CA bundle/trust changes can reject formerly trusted chains. The actual Stooq request completed TLS and reached HTTP 404. [Upstream release](https://github.com/certifi/python-certifi/releases/tag/2026.10.18). |
| Flask | 3.1.2 → 3.1.3 | Patch changes session-access tracking. RiskWise does not use Flask session storage; Dash routes/callbacks passed. [Release notes](https://flask.palletsprojects.com/en/stable/changes/#version-3-1-3). |
| idna | 3.11 → 3.15 | Earlier DNS-label length rejection and internal changes. RiskWise uses a fixed valid ASCII hostname, not custom IDNA handling. [Release notes](https://github.com/kjd/idna/releases/tag/v3.15). |
| Pillow | 12.1.1 → 13.0.0 | Major release drops Python 3.10 and removes deprecated image APIs. No direct Pillow API use in project source; Python 3.12 and the Matplotlib PNG smoke path passed. [Breaking changes](https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html#backwards-incompatible-changes). |
| requests | 2.32.5 → 2.33.0 | Drops Python 3.9; changes zipped-path extraction and malformed netrc handling. RiskWise uses ordinary HTTPS GET with a timeout. [Release notes](https://requests.readthedocs.io/en/latest/community/updates/). |
| setuptools | 82.0.0 → 84.1.0 | Python ≥3.10, compiler/distutils changes and stricter archive extraction. No project setup script or custom extension build; wheel-only resolution avoids source compiler paths. [Release history](https://setuptools.pypa.io/en/latest/history.html). |
| urllib3 | 2.6.3 → 2.8.0 | Stricter host validation and HTTPS proxy TLS behavior. RiskWise defines no custom adapters/proxy TLS configuration; private deployment proxy settings were not inspected. [Changelog](https://urllib3.readthedocs.io/en/stable/changelog.html). |
| Werkzeug | 3.1.5 → 3.1.9 | Host/port, authentication, path and request-parser hardening. Valid local Dash requests passed; reverse-proxy behavior remains a hosted check. [Release notes](https://werkzeug.palletsprojects.com/en/stable/changes/). |

## Deployment and external services

Render requires listening on `0.0.0.0`; the installed Gunicorn 25.0.3 configuration defaults to `0.0.0.0:$PORT` when `PORT` is set. This supports compatibility of the unchanged Procfile with the documented Render environment. [Render port requirements](https://render.com/docs/web-services#port-binding).

No Docker/WSL Linux execution environment was available. The new CI workflow is structurally consistent with the local tests and starts Gunicorn on Ubuntu, but it has not run on GitHub. Private Render runtime/build/start/auto-deploy settings were not accessed. Python 3.12 is the verified setup target; wheel resolution is not proof of an actual Render build or worker startup.

The existing public demo returned HTTP 503 with a **SERVICE WAKING UP** page during this review. A later bounded retry returned HTTP 200 and rendered both charts. This is an observed cold-start availability issue in the existing deployment, not a deployment test of the local changes.

The Stooq `spy.us` CSV endpoint returned HTTP 404. Live historical calibration could not be verified. Both the local browser and unit tests confirm fallback to preset assumptions, now honestly labelled. The README records this limitation. Do not advertise that live calibration or uninterrupted demo availability has been validated.

## Remaining risks and publication boundary

The known log-return/simple-return approximation, different deterministic rate convention, illustrative crash haircut affecting later contributions, finite-sample solver and theoretical normal returns below −100% remain documented and unchanged. There are no server-enforced computation caps, load-test evidence, database/accounts or production observability. Browser-local plans retain their existing schema and storage behavior.

No secrets or sensitive candidate files were found. `.venv`, environment files, bytecode, coverage and lint caches are ignored; review environments, scan JSON, comparison scripts and exported test plans are outside the repository. Pattern scans and vulnerability databases cannot guarantee absence of every possible secret or vulnerability.

Publish the source with those limits stated. Require a green hosted workflow and review of Render's actual settings before treating a later rollout as validated. A future model correction or replacement data provider should be a separate, explicitly tested change.

Suggested commit subject: `Improve RiskWise numerical validation, solver safety, CI and documentation`
