# Engineering audit and validation

Audited 3 November 2026 against upstream commit `2cb24bc` (`Update README.md`). Changes are local and uncommitted. The existing project was cloned into `RiskWise-Financial-Risk-Simulator` inside the supplied workspace, whose root already contained an empty Git repository. No unrelated project was created.

See the [final engineering review](FINAL_REVIEW.md) for the subsequent independent comparisons, expanded 42-test suite, dependency compatibility review and latest external-service checks. The initial audit below recorded 34 tests and an earlier successful live-demo check.

## Findings and improvements

| Area | Original finding | Outcome |
| --- | --- | --- |
| Architecture | One Dash/Flask app with numerical/data helpers; most projection/search logic in `app.py` | Extracted `projections.py` and `planning.py` for independent testing; preserved `app:server` entrypoint |
| Numerical model | Monthly independent normal simple returns; fixed nominal contributions and real-value discounting | Preserved model and original seeded results; documented formulas and approximations |
| Randomness | Every simulation reset global NumPy RNG state | Local legacy RNG gives identical draws without shared-state interference |
| Solver | Always returned a recommendation even for infeasible goals; nearest-£10 rounding could undershoot | Checks zero and upper bounds, reports infeasibility, rounds feasible bracket upward |
| Solver UI | Results could become stale when settings changed; calibration fallback was silent | After Solve, relevant settings trigger recalculation; fallback is disclosed; initial render waits for a click |
| Inputs | Invalid core parameters could create empty/non-finite output or invalid arithmetic | Added finite/domain checks for simulation, statistics, calibration and crash severity; zero-horizon crash keeps initial balance |
| Tests | No test suite | Added 34 unittest cases covering numerical behavior, edge cases, data failures and application integration |
| CI | No tracked Actions workflow despite README claiming GitHub CI/CD | Added tests, compile/dependency checks, Ubuntu WSGI smoke test and vulnerability audit; no deployment claim |
| Dependencies | Audit reported known advisories affecting eight pinned packages | Updated affected pins; no new application dependencies; repeat audit found no known vulnerabilities |
| Repository hygiene | Three tracked CPython 3.14 bytecode files; no `.gitignore`; UTF-16 requirements file | Removed tracked bytecode from working tree; added ignores; requirements now readable UTF-8 |
| Documentation | Missing setup/tests; model detail and deployment claims incomplete | Concise README, real architecture diagram, genuine screenshots and separate methodology/development notes |

`SRC/main.py` remains the interactive Matplotlib CLI. `SRC/checkdata.py`, `SRC/plots.py` and `SRC/init.py` are legacy diagnostics/helpers outside the deployed entrypoint. They are not presented as a packaged library or separate backend service.

## Validation evidence

| Check | Result |
| --- | --- |
| Fresh isolated Python 3.12.14 environment | Original pins installed; updated pins installed successfully |
| `PYTHONPATH=SRC python -m unittest discover -s tests -v` | 34 tests passed |
| `python -m compileall -q SRC` | Passed |
| `python -m pip check` | No broken requirements |
| `pip-audit -r requirements.txt --no-deps --disable-pip` | No known vulnerabilities in updated pins at audit time |
| Flask test client | `/`, `/_dash-layout`, `/_dash-dependencies` returned 200 |
| Updated local app in headless Edge | Dashboard charts, solver recalculation/infeasibility, plan save/JSON download, sequence demo and method tab passed; no JavaScript or HTTP 5xx errors |
| Existing public Render demo | HTTP 200, title RiskWise, two charts rendered, no browser JavaScript errors |
| Screenshots | Local dashboard and sequence demo captured from the running app and visually inspected; not mockups |
| Credentials | Pattern scan across text files in all 19 fetched revisions found no matches; sensitive/generated filename review found bytecode but no tracked environment or key files |
| Documentation links | Local Markdown targets checked; public demo and upstream repository accessible |

Dependency changes: certifi 2026.1.4 → 2026.10.18; Flask 3.1.2 → 3.1.3; idna 3.11 → 3.15; Pillow 12.1.1 → 13.0.0; requests 2.32.5 → 2.33.0; setuptools 82.0.0 → 84.1.0; urllib3 2.6.3 → 2.8.0; Werkzeug 3.1.5 → 3.1.9. The original audit reported 95 advisory entries, some overlapping, across these eight packages; that is not a count of exploitable application vulnerabilities. Scans are point-in-time checks, not security certification.

The live demo runs the previously deployed code, not these local changes. Its accessibility does not establish that private Render auto-deployment settings are correct. Gunicorn cannot run natively in this Windows environment, so the added Linux CI smoke test and hosted Actions run remain pending a future push. The Procfile is unchanged; no deployment, commit or push was performed.

## Remaining credibility limits

1. **Calibration and rate conventions:** historical mean log returns feed a simple-return engine; the deterministic overlay uses a different monthly conversion. Do not describe the overlay as the simulator's expected value or claim validated forecasting accuracy.
2. **Crash model:** the post-processing haircut also reduces later contributions. It is illustrative stress, not a faithful one-month market shock. This was documented rather than silently changing existing results.
3. **Normal-return assumptions:** no fat tails, correlations or regime changes. Returns below −100% are theoretically possible. P10/P90 are sampled quantiles, not worst-case bounds or confidence intervals.
4. **Comparisons and solver:** calibrated scenarios can differ only by random sampling; sensitivity uses different seeds; the solver uses its own 900-path sample and assumes monotonicity. Its target is conditional on assumptions, not guaranteed success.
5. **Operational scope:** synchronous callbacks, process-local cache, browser-only plans, no server-side resource caps or load-test evidence. This is a working educational web application, not a production trading platform or distributed backend.
6. **External services:** Stooq and the Bootstrap CDN remain availability dependencies. Market-data unit tests use mocks; historical data accuracy and service reliability are not certified.

The highest-value follow-up would be an explicitly versioned model revision with consistent return conventions and crash application inside the recurrence, supported by new regression tests. That would intentionally change outputs and should be reviewed separately.
