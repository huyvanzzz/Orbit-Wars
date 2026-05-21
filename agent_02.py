import math
import os

os.environ["KAGGLE_ENVELOPES"] = "0"


def _unpack(obs):
    if isinstance(obs, dict):
        return (
            obs.get("player", 0),
            obs.get("planets", []),
            obs.get("fleets", []),
            obs.get("angular_velocity", 0.0),
        )
    return (
        getattr(obs, "player", 0),
        getattr(obs, "planets", []),
        getattr(obs, "fleets", []),
        getattr(obs, "angular_velocity", 0.0),
    )


def agent(obs):
    player, planets_data, _, _ = _unpack(obs)
    planets = [p[:7] for p in planets_data]
    my_planets = [p for p in planets if p[1] == player]
    neutral_planets = [p for p in planets if p[1] == -1]

    if not my_planets:
        return []

    moves = []
    for src in my_planets:
        if not neutral_planets:
            continue
        target = min(neutral_planets, key=lambda t: math.hypot(src[2] - t[2], src[3] - t[3]))
        if src[5] > target[5] + 3:
            angle = math.atan2(target[3] - src[3], target[2] - src[2])
            ships = max(1, int(src[5] * 0.35))
            moves.append([src[0], angle, ships])

    return moves