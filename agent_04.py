import math
import os

os.environ["KAGGLE_ENVELOPES"] = "0"


def _unpack(obs):
    if isinstance(obs, dict):
        return obs.get("player", 0), obs.get("planets", []), obs.get("fleets", []), obs.get("angular_velocity", 0.0)
    return getattr(obs, "player", 0), getattr(obs, "planets", []), getattr(obs, "fleets", []), getattr(obs, "angular_velocity", 0.0)


def _distance(a, b):
    return math.hypot(a[2] - b[2], a[3] - b[3])


def agent(obs):
    player, planets_data, fleets_data, _ = _unpack(obs)
    planets = [p[:7] for p in planets_data]
    fleets = [f[:7] for f in fleets_data]

    my_planets = [p for p in planets if p[1] == player]
    if not my_planets:
        return []

    enemy_fleets = [f for f in fleets if f[1] != player]
    neutrals = [p for p in planets if p[1] == -1]
    enemies = [p for p in planets if p[1] not in (-1, player)]

    moves = []
    for src in sorted(my_planets, key=lambda p: p[5], reverse=True):
        threatened = None
        if enemy_fleets:
            threatened = min(enemy_fleets, key=lambda f: math.hypot(src[2] - f[2], src[3] - f[3]))

        if threatened is not None and src[5] > 20:
            angle = math.atan2(src[3] - threatened[3], src[2] - threatened[2])
            moves.append([src[0], angle, max(5, int(src[5] * 0.3))])
            continue

        target_pool = neutrals if neutrals else enemies
        if not target_pool:
            continue

        target = min(target_pool, key=lambda t: (_distance(src, t), -t[6]))
        cost = target[5] + (4 if target[1] == -1 else 8)
        if src[5] > cost:
            angle = math.atan2(target[3] - src[3], target[2] - src[2])
            moves.append([src[0], angle, int(cost) + 1])

    return moves