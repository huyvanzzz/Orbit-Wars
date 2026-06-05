# Orbit Wars Workflow

## Files

- `prompt.md`: project-specific instructions and stop conditions.
- `main.py`: active submission agent unless the user says otherwise.
- `abc.py` and `agent_*.py`: comparison/reference agents; read-only by default.
- `orbit_wars_benchmark.ipynb`: source of true match order and evaluation shape.
- `replays/last_match.html`: latest visual replay; interpret with the notebook slot order.
- `json/`: replay or match traces when available.
- `tune_grid_search.py`: parameter search and `main.py` patching utility.
- `orbit-wars.md`: project notes/version log when present.

## Iteration Pattern

1. Establish baseline from the current notebook and current `main.py`.
2. Inspect failed or narrow wins in replay artifacts.
3. Identify a strategic weakness, for example:
   - early expansion too passive or too risky
   - over-defending and holding ships too long
   - attacking the wrong opponent in FFA
   - misreading leader pressure
   - combined attacks draining too many ships
   - relaunch/handoff behavior causing instability
4. Change logic with a bounded fallback.
5. Screen quickly, then confirm if promising.

## Match Counts

- Use 3-5 matches for cheap candidate screening.
- Use 10 matches for a candidate that looks promising.
- Use 20-25 matches for confirmation when the user wants stronger evidence.
- If the user specifies a threshold, follow it exactly.

## Robustness

When randomness and agent order matter:

- Run at least two normal random batches.
- Run at least one batch with a changed agent order.
- Always map results back to the active agent slot.

## Parameter Tuning

Do not start with parameter-only tuning when the user asks for broad improvement. First create a logic candidate. Then use `tune_grid_search.py` to tune thresholds around that logic.

After selecting an index:

```powershell
.\.venv\Scripts\python.exe tune_grid_search.py --apply-index <idx>
```

This writes the chosen config directly into `main.py`.
