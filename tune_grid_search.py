import argparse
import csv
import json
import logging
import os
import re
import statistics
import time
from collections import defaultdict
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


ROOT = Path(__file__).resolve().parent
MAIN_PATH = ROOT / "main.py"
RESULTS_DIR = ROOT / "grid_results"
REPLAY_DIR = ROOT / "replays"

# Same order as the current orbit_wars_benchmark.ipynb:
# slot0 hellburner_ref, slot1 main_plus, slot2 main, slot3 agent_04.
AGENT_FILES = [
    ROOT / "hellburner_ref.py",
    ROOT / "main_plus_main5_ideas.py",
    ROOT / "main.py",
    ROOT / "agent_04.py",
]

os.environ.setdefault("KAGGLE_ENVELOPES", "0")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("LITELLM_LOG", "ERROR")
logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

from kaggle_environments import make


def build_param_grid():
    base = {
        "search_pick": 5,
        "defense_oversend_4p": 0,
        "leader_ratio": 1.10,
        "leader_bonus": 10.0,
        "leader_min_step": 40,
        "focus_enabled": True,
        "focus_turn": 65,
        "focus_gap": 12,
        "weakest_enabled": False,
        "hammer_turn": 0,
        "frontier_enabled": False,
        "frontier_turn_min": 70,
        "frontier_turn_max": 170,
        "frontier_dist": 26.0,
        "frontier_base": 10,
        "frontier_prod_mult": 3,
        "frontier_frac": 0.25,
        "pressure_enabled": False,
        "pressure_turn": 35,
        "pressure_max_moves": 1,
        "pressure_min_avail": 28,
        "pressure_leader_bonus": 28.0,
        "pressure_enemy_bonus": 8.0,
        "pressure_neutral_penalty": 8.0,
        "acc_lead": 100,
        "acc_feed_min": 30,
        "acc_keep": 30,
        "acc_max_feeds": 3,
        "brain_enabled": True,
        "brain_min": 200,
        "brain_require_target": False,
        "brain_frontier": False,
        "brain_frontier_weight": 2.0,
        "mega_min": 300,
        "mega_thresholds": "{5: 200, 4: 250, 3: 300, 2: 350, 1: 400}",
        "mega_target_cap": 100,
    }

    variations = [
        {},
        {"focus_turn": 60},
        {"focus_turn": 70},
        {"focus_turn": 75},
        {"focus_gap": 10},
        {"focus_gap": 14},
        {"leader_bonus": 8.0},
        {"leader_bonus": 12.0},
        {"leader_min_step": 35},
        {"leader_min_step": 50},
        {"leader_ratio": 1.05},
        {"leader_ratio": 1.15},
        {"search_pick": 4},
        {"search_pick": 6},
        {"defense_oversend_4p": 1},
        {"weakest_enabled": True},
        {"hammer_turn": 35},
        {"hammer_turn": 55},
        {"acc_lead": 70, "acc_feed_min": 22, "acc_keep": 24, "brain_min": 130},
        {"acc_lead": 85, "acc_feed_min": 26, "acc_keep": 28, "brain_min": 160},
        {"acc_lead": 120, "acc_feed_min": 34, "acc_keep": 34, "brain_min": 220},
        {"acc_max_feeds": 2},
        {"acc_lead": 70, "acc_feed_min": 22, "acc_keep": 24, "brain_min": 130, "acc_max_feeds": 2},
        {"brain_enabled": False},
        {"brain_require_target": True},
        {"brain_frontier": True, "brain_frontier_weight": 1.0},
        {"brain_frontier": True, "brain_frontier_weight": 2.0},
        {"brain_frontier": True, "brain_frontier_weight": 3.0},
        {"mega_min": 260, "mega_thresholds": "{5: 160, 4: 200, 3: 240, 2: 300, 1: 380}", "mega_target_cap": 110},
        {"mega_min": 220, "mega_thresholds": "{5: 120, 4: 150, 3: 190, 2: 240, 1: 320}", "mega_target_cap": 120},
        {"frontier_enabled": True, "frontier_turn_min": 60, "frontier_turn_max": 150, "frontier_dist": 20.0, "frontier_base": 4, "frontier_prod_mult": 2, "frontier_frac": 0.10},
        {"frontier_enabled": True, "frontier_turn_min": 70, "frontier_turn_max": 160, "frontier_dist": 22.0, "frontier_base": 6, "frontier_prod_mult": 2, "frontier_frac": 0.14},
        {"pressure_enabled": True, "pressure_turn": 60, "pressure_max_moves": 4, "pressure_min_avail": 35, "pressure_leader_bonus": 45.0, "pressure_enemy_bonus": 0.0, "pressure_neutral_penalty": 30.0},
        {"pressure_enabled": True, "pressure_turn": 70, "pressure_max_moves": 1, "pressure_min_avail": 45, "pressure_leader_bonus": 50.0, "pressure_enemy_bonus": -4.0, "pressure_neutral_penalty": 35.0},
        {"focus_turn": 60, "focus_gap": 10, "leader_bonus": 8.0},
        {"focus_turn": 70, "focus_gap": 12, "leader_bonus": 8.0},
        {"focus_turn": 65, "focus_gap": 14, "leader_bonus": 12.0},
        {"leader_ratio": 1.15, "leader_min_step": 50, "leader_bonus": 8.0},
        {"leader_ratio": 1.05, "leader_min_step": 35, "leader_bonus": 12.0},
        {"search_pick": 4, "focus_turn": 60, "focus_gap": 10, "leader_bonus": 8.0},
        {"search_pick": 6, "focus_turn": 70, "focus_gap": 12},
        {"acc_lead": 70, "acc_feed_min": 22, "acc_keep": 24, "brain_min": 130, "brain_require_target": True},
        {"acc_lead": 85, "acc_feed_min": 26, "acc_keep": 28, "brain_min": 160, "brain_require_target": True},
        {"acc_lead": 70, "acc_feed_min": 22, "acc_keep": 24, "brain_min": 130, "mega_min": 260, "mega_thresholds": "{5: 150, 4: 190, 3: 240, 2: 300, 1: 380}", "mega_target_cap": 110},
        {"acc_lead": 85, "acc_feed_min": 26, "acc_keep": 28, "brain_min": 160, "mega_min": 260, "mega_thresholds": "{5: 160, 4: 200, 3: 240, 2: 300, 1: 380}", "mega_target_cap": 110},
        {"frontier_enabled": True, "frontier_turn_min": 70, "frontier_turn_max": 150, "frontier_dist": 18.0, "frontier_base": 5, "frontier_prod_mult": 3, "frontier_frac": 0.12},
        {"frontier_enabled": True, "frontier_turn_min": 60, "frontier_turn_max": 150, "frontier_dist": 20.0, "frontier_base": 4, "frontier_prod_mult": 2, "frontier_frac": 0.10, "acc_lead": 85, "brain_min": 160},
        {"pressure_enabled": True, "pressure_turn": 60, "pressure_max_moves": 2, "pressure_min_avail": 40, "pressure_leader_bonus": 40.0, "pressure_enemy_bonus": 0.0, "pressure_neutral_penalty": 30.0},
        {"pressure_enabled": True, "pressure_turn": 65, "pressure_max_moves": 1, "pressure_min_avail": 35, "pressure_leader_bonus": 38.0, "pressure_enemy_bonus": 4.0, "pressure_neutral_penalty": 20.0},
        {"focus_enabled": False, "leader_ratio": 1.20, "leader_bonus": 10.0, "leader_min_step": 45},
        {"focus_enabled": False, "leader_ratio": 1.10, "leader_bonus": 12.0, "leader_min_step": 45},
        {"focus_enabled": False, "pressure_enabled": True, "pressure_turn": 60, "pressure_max_moves": 4, "pressure_min_avail": 35, "pressure_leader_bonus": 45.0, "pressure_enemy_bonus": 0.0, "pressure_neutral_penalty": 30.0},
        {"weakest_enabled": True, "focus_turn": 70, "focus_gap": 14, "leader_bonus": 8.0},
        {"weakest_enabled": True, "leader_ratio": 1.20, "leader_bonus": 6.0, "leader_min_step": 55},
        {"defense_oversend_4p": 1, "frontier_enabled": True, "frontier_base": 4, "frontier_prod_mult": 2, "frontier_frac": 0.10},
        {"defense_oversend_4p": 1, "acc_lead": 85, "acc_feed_min": 26, "acc_keep": 28, "brain_min": 160},
        {"mega_min": 260, "mega_target_cap": 120, "leader_bonus": 12.0},
        {"mega_min": 220, "mega_target_cap": 120, "acc_lead": 100, "brain_min": 200},
        {"hammer_turn": 45, "leader_ratio": 1.15, "leader_bonus": 8.0},
        {"search_pick": 6, "acc_lead": 70, "acc_feed_min": 22, "acc_keep": 24, "brain_min": 130, "focus_turn": 70, "focus_gap": 12},
    ]

    configs = []
    for variation in variations[:60]:
        cfg = dict(base)
        cfg.update(variation)
        configs.append(cfg)
    return configs


def py_bool(value):
    return "True" if value else "False"


def replace_assign(text, name, value):
    pattern = rf"^{name}\s*=.*$"
    replacement = f"{name} = {value}"
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Could not patch {name}; matches={count}")
    return new_text


def patch_main(original_text, cfg):
    text = original_text
    assignments = {
        "SEARCH_MAX_ACTIONS_TO_PICK": cfg["search_pick"],
        "DEFENSE_OVERSEND_4P": cfg["defense_oversend_4p"],
        "WEAKEST_TARGET_ENABLED": py_bool(cfg["weakest_enabled"]),
        "LEADER_BASH_RATIO": cfg["leader_ratio"],
        "LEADER_BASH_BONUS": cfg["leader_bonus"],
        "LEADER_BASH_MIN_STEP": cfg["leader_min_step"],
        "FOUR_P_LEADER_FOCUS_ENABLED": py_bool(cfg["focus_enabled"]),
        "FOUR_P_LEADER_FOCUS_TURN_MIN": cfg["focus_turn"],
        "FOUR_P_LEADER_FOCUS_PROD_GAP": cfg["focus_gap"],
        "HAMMER_4P_TURN_MIN": cfg["hammer_turn"],
        "FRONTIER_GUARD_ENABLED": py_bool(cfg["frontier_enabled"]),
        "FRONTIER_GUARD_TURN_MIN": cfg["frontier_turn_min"],
        "FRONTIER_GUARD_TURN_MAX": cfg["frontier_turn_max"],
        "FRONTIER_GUARD_ENEMY_DIST": cfg["frontier_dist"],
        "FRONTIER_GUARD_BASE": cfg["frontier_base"],
        "FRONTIER_GUARD_PROD_MULT": cfg["frontier_prod_mult"],
        "FRONTIER_GUARD_FRAC": cfg["frontier_frac"],
        "FOUR_P_PRESSURE_FALLBACK_ENABLED": py_bool(cfg["pressure_enabled"]),
        "FOUR_P_PRESSURE_TURN_MIN": cfg["pressure_turn"],
        "FOUR_P_PRESSURE_MAX_MOVES_BEFORE": cfg["pressure_max_moves"],
        "FOUR_P_PRESSURE_MIN_AVAIL": cfg["pressure_min_avail"],
        "FOUR_P_PRESSURE_LEADER_BONUS": cfg["pressure_leader_bonus"],
        "FOUR_P_PRESSURE_ENEMY_BONUS": cfg["pressure_enemy_bonus"],
        "FOUR_P_PRESSURE_NEUTRAL_PENALTY": cfg["pressure_neutral_penalty"],
        "ACCUMULATOR_LEAD_MIN_SHIPS": cfg["acc_lead"],
        "ACCUMULATOR_FEEDER_MIN_SURPLUS": cfg["acc_feed_min"],
        "ACCUMULATOR_FEEDER_KEEP_RESERVE": cfg["acc_keep"],
        "ACCUMULATOR_MAX_FEEDS_PER_TURN": cfg["acc_max_feeds"],
        "BRAIN_LEAD_RESERVE_ENABLED": py_bool(cfg["brain_enabled"]),
        "BRAIN_LEAD_RESERVE_MIN_SHIPS": cfg["brain_min"],
        "BRAIN_LEAD_RESERVE_REQUIRE_TARGET": py_bool(cfg["brain_require_target"]),
        "BRAIN_LEAD_PREFER_FRONTIER": py_bool(cfg["brain_frontier"]),
        "BRAIN_LEAD_FRONTIER_WEIGHT": cfg["brain_frontier_weight"],
        "MEGA_HAMMER_SHIPS_MIN": cfg["mega_min"],
        "MEGA_HAMMER_THRESHOLD_BY_PROD": cfg["mega_thresholds"],
        "MEGA_HAMMER_TARGET_GARRISON_MAX_ITER_H": cfg["mega_target_cap"],
    }
    for name, value in assignments.items():
        text = replace_assign(text, name, value)
    return text


def load_agent_from_file(file_path):
    file_path = Path(file_path).resolve()
    unique = f"{time.time_ns()}_{abs(hash(str(file_path))) & 0xffffffff}"
    module_name = f"orbit_wars_{file_path.stem}_{unique}"
    loader = SourceFileLoader(module_name, str(file_path))
    spec = spec_from_loader(module_name, loader)
    module = module_from_spec(spec)
    loader.exec_module(module)
    if not hasattr(module, "agent"):
        raise AttributeError(f"{file_path} must define agent(obs)")
    return module.agent


def play_match(agent_files, debug=False):
    env = make("orbit_wars", debug=debug)
    env.run([load_agent_from_file(path) for path in agent_files])
    final_states = env.steps[-1]
    rewards = [state.reward for state in final_states]
    statuses = [state.status for state in final_states]
    return env, rewards, statuses


def save_replay_html(env, name="last_match", width=900, height=650):
    REPLAY_DIR.mkdir(exist_ok=True)
    replay_path = REPLAY_DIR / f"{name}.html"
    replay_path.write_text(env.render(mode="html", width=width, height=height), encoding="utf-8")
    print("Saved replay:", replay_path.resolve())


def evaluate_config(matches):
    scoreboard = defaultdict(list)
    statuses_seen = []
    last_env = None
    for _ in range(matches):
        last_env, rewards, statuses = play_match(AGENT_FILES, debug=False)
        statuses_seen.append(statuses)
        for file_path, reward in zip(AGENT_FILES, rewards):
            scoreboard[file_path.stem].append(reward)

    main_vals = scoreboard["main"]
    result = {
        "matches": matches,
        "main_avg": sum(main_vals) / len(main_vals),
        "main_wins": main_vals.count(1),
        "main_losses": main_vals.count(-1),
        "main_median": statistics.median(main_vals),
        "all_done": all(all(status == "DONE" for status in statuses) for statuses in statuses_seen),
    }
    for file_path in AGENT_FILES:
        name = file_path.stem
        vals = scoreboard[name]
        result[f"{name}_avg"] = sum(vals) / len(vals)
        result[f"{name}_wins"] = vals.count(1)
    return result, last_env


def write_results(rows, stamp):
    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / f"grid_results_{stamp}.json"
    csv_path = RESULTS_DIR / f"grid_results_{stamp}.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return csv_path, json_path


def print_config(index, cfg):
    print(json.dumps({"index": index, **cfg}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=3, help="Matches per config.")
    parser.add_argument("--limit", type=int, default=60, help="Number of configs to test.")
    parser.add_argument("--start", type=int, default=0, help="Start index for resuming.")
    parser.add_argument("--indices", default="", help="Comma-separated config indices, e.g. 0,12,27.")
    parser.add_argument("--show-index", type=int, default=None, help="Print one config and exit.")
    parser.add_argument("--apply-index", type=int, default=None, help="Patch main.py permanently and exit.")
    parser.add_argument("--no-replay", action="store_true", help="Do not save replays/last_match.html.")
    args = parser.parse_args()

    original_text = MAIN_PATH.read_text(encoding="utf-8")
    all_configs = build_param_grid()

    if args.show_index is not None:
        print_config(args.show_index, all_configs[args.show_index])
        return

    if args.apply_index is not None:
        cfg = all_configs[args.apply_index]
        MAIN_PATH.write_text(patch_main(original_text, cfg), encoding="utf-8")
        print_config(args.apply_index, cfg)
        print("Applied to main.py")
        return

    if args.indices:
        indices = [int(part.strip()) for part in args.indices.split(",") if part.strip()]
        configs = [(idx, all_configs[idx]) for idx in indices]
    else:
        configs = list(enumerate(all_configs[args.start : args.start + args.limit], start=args.start))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    rows = []
    last_env = None
    iterator = tqdm(configs, total=len(configs), desc="grid search") if tqdm else configs

    try:
        for index, cfg in iterator:
            MAIN_PATH.write_text(patch_main(original_text, cfg), encoding="utf-8")
            result, last_env = evaluate_config(args.matches)
            row = {"index": index, **cfg, **result}
            rows.append(row)
            rows.sort(key=lambda item: (item["main_wins"], item["main_avg"]), reverse=True)
            write_results(rows, stamp)
            best = rows[0]
            msg = f"best idx={best['index']} wins={best['main_wins']}/{best['matches']} avg={best['main_avg']:.3f}"
            if tqdm:
                iterator.set_postfix_str(msg)
            else:
                print(f"idx={index}: wins={row['main_wins']}/{args.matches} avg={row['main_avg']:.3f}; {msg}")
    finally:
        MAIN_PATH.write_text(original_text, encoding="utf-8")

    csv_path, json_path = write_results(rows, stamp)
    print("Saved CSV:", csv_path)
    print("Saved JSON:", json_path)
    if last_env is not None and not args.no_replay:
        save_replay_html(last_env)
    if rows:
        print("Top 10:")
        for row in rows[:10]:
            print(
                f"idx={row['index']:02d} wins={row['main_wins']}/{row['matches']} "
                f"avg={row['main_avg']:.3f} focus={row['focus_turn']}/{row['focus_gap']} "
                f"bash={row['leader_ratio']}/{row['leader_bonus']} "
                f"acc={row['acc_lead']}/{row['acc_feed_min']}/{row['brain_min']} "
                f"pressure={row['pressure_enabled']}"
            )


if __name__ == "__main__":
    main()
