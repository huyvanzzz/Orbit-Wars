import json
import sys
from pathlib import Path


REQUIRED = [
    "prompt.md",
    "main.py",
    "orbit_wars_benchmark.ipynb",
]

OPTIONAL = [
    "abc.py",
    "tune_grid_search.py",
    "replays/last_match.html",
    "json",
    "orbit-wars.md",
    "AGENTS.md",
]


def main() -> int:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
    missing = [name for name in REQUIRED if not (root / name).exists()]
    present_optional = [name for name in OPTIONAL if (root / name).exists()]

    notebook_agents = []
    nb_path = root / "orbit_wars_benchmark.ipynb"
    if nb_path.exists():
        try:
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
            text = "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))
            for candidate in ["main.py", "abc.py", "agent_02.py", "agent_03.py", "agent_04.py"]:
                if candidate in text:
                    notebook_agents.append(candidate)
        except Exception as exc:
            print(f"Could not inspect notebook: {exc}")

    print(f"Root: {root}")
    print(f"Required missing: {missing if missing else 'none'}")
    print(f"Optional present: {', '.join(present_optional) if present_optional else 'none'}")
    print(f"Notebook mentions: {', '.join(notebook_agents) if notebook_agents else 'none detected'}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
