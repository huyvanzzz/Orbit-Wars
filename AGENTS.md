# Agent Coding Guide

## Scope

This repo is an Orbit Wars agent optimization workspace. Follow `prompt.md` first, then this file.

## Active Agent

- `main.py` is the active submission agent unless the user explicitly changes it.
- Read `abc.py`, `agent_02.py`, `agent_03.py`, and `agent_04.py` as references only.
- Do not modify comparison agents unless the user asks.

## Benchmark Workflow

- Read `orbit_wars_benchmark.ipynb` before interpreting results; the agent order in the notebook defines replay slots.
- Current grid-search helper expects the notebook-style order in `tune_grid_search.py`.
- Use `.venv\Scripts\python.exe` for local scripts in this repo.

## Candidate Thresholds

- Use 3-5 matches only to reject weak candidates quickly.
- If a candidate looks good, run at least 10 matches.
- For stronger confirmation, run 20-25 matches.
- If the user gives a threshold such as `10/20` or `15/30`, use that threshold.

## Replay Analysis

- Inspect `replays/last_match.html` after benchmark runs.
- Remember that the HTML shows slots; map slots back to the notebook agent order.
- Use `json/` traces when available to identify why `main.py` won or lost.

## Grid Search

Run:

```powershell
.\.venv\Scripts\python.exe tune_grid_search.py --matches 3 --limit 60
```

Apply a selected index directly into `main.py`:

```powershell
.\.venv\Scripts\python.exe tune_grid_search.py --apply-index <idx>
```

After applying, `main.py` contains the chosen parameters and does not need an external config file.

## Notes

- Prefer broad behavioral improvements before parameter tuning.
- Update `orbit-wars.md` when making meaningful versions or recording failed directions.
- Do not claim a result unless it was run in the current session.
