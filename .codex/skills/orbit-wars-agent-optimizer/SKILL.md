---
name: orbit-wars-agent-optimizer
description: Iterative workflow for improving Orbit Wars Kaggle agents in a coding repo. Use when Codex is asked to read prompt.md, main.py, abc.py, benchmark notebooks, replay HTML/JSON, tune_grid_search.py, or otherwise improve, benchmark, grid-search, validate, or package an Orbit Wars agent without stopping after a single edit.
---

# Orbit Wars Agent Optimizer

## Core Rule

Optimize by evidence, not by one-off parameter tweaks. Read the current project state, form a hypothesis, edit only the active agent file unless asked otherwise, benchmark, inspect replay artifacts, and iterate until the requested threshold is reached or a concrete blocker is found.

## Startup

1. Read `prompt.md`, `AGENTS.md` if present, `orbit_wars_benchmark.ipynb`, and the active `main.py`.
2. Confirm the notebook's actual agent order before interpreting rewards or `replays/last_match.html`; do not assume `main.py` is always in slot 0.
3. Read comparison agents such as `abc.py`, `agent_02.py`, `agent_03.py`, and `agent_04.py` for ideas, but do not modify them unless the user explicitly asks.
4. If `tune_grid_search.py` exists, inspect it before changing parameters or applying a config.
5. Run `scripts/check_orbit_wars_project.py <repo>` from this skill when you need a quick project sanity check.

## Improvement Loop

Repeat this loop for real optimization work:

1. Diagnose: inspect code, recent benchmark output, `json/` replays, and `replays/last_match.html`.
2. Hypothesize: state the behavioral weakness and the code area to change.
3. Implement: make a scoped logic change in `main.py`; avoid broad rewrites unless the evidence supports them.
4. Screen: run 3-5 matches for weak candidate detection.
5. Confirm: when a candidate is promising, run at least 10 matches; for stronger claims use 20-25 matches.
6. Replay: inspect the newest replay and map slot/player positions to the notebook order.
7. Record: update the project notes file if present, especially version changes and weaker attempts.
8. Continue: if below threshold, use the replay failure mode to adjust logic or constraints; do not stop at rollback.

## Candidate Rules

- Treat very weak 3-5 match screens as rejectable.
- Treat noisy wins as provisional; rerun promising candidates with random seeds and at least one changed agent order when the user asks for robustness.
- Use the user's threshold when given. In this project, common thresholds are above 5/10, at least 10/20, or about 15/30 depending on the phase.
- Prefer broad behavioral improvements before grid search. Use grid search after a logic candidate is plausible.

## Grid Search

Use `tune_grid_search.py` when the user asks for parameter search or applying a config.

Common commands:

```powershell
.\.venv\Scripts\python.exe tune_grid_search.py --matches 3 --limit 60
.\.venv\Scripts\python.exe tune_grid_search.py --matches 6 --indices 27,24,32
.\.venv\Scripts\python.exe tune_grid_search.py --show-index 27
.\.venv\Scripts\python.exe tune_grid_search.py --apply-index 27
```

After `--apply-index`, `main.py` contains the parameters directly and is submission-ready, assuming no other validation fails.

## Project Setup

When asked to standardize the repo for coding agents, create or update `AGENTS.md` with:

- active file and files that must not be edited
- benchmark commands
- candidate thresholds
- replay interpretation rules
- grid-search apply workflow
- notes/update expectations

Keep setup files concise and operational. Do not add broad documentation that future agents will ignore.

## References

- Read `references/orbit-wars-workflow.md` for detailed workflow and thresholds.
- Read `references/project-setup.md` when creating or updating `AGENTS.md`.
