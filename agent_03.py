import math
import os

os.environ["KAGGLE_ENVELOPES"] = "0"


def _unpack(obs):
    if isinstance(obs, dict):
        return (
            obs.get("player", 0),
            obs.get("planets", []),
            obs.get("fleets", []),
        )
    return (
        getattr(obs, "player", 0),
        getattr(obs, "planets", []),
        getattr(obs, "fleets", []),
    )


def agent(obs):
    player, planets_data, _ = _unpack(obs)
    planets = [p[:7] for p in planets_data]
    my_planets = [p for p in planets if p[1] == player]
    enemy_planets = [p for p in planets if p[1] not in (-1, player)]

    if not my_planets:
        return []

    moves = []
    for src in sorted(my_planets, key=lambda p: p[5], reverse=True):
        if enemy_planets:
            target = min(enemy_planets, key=lambda t: (t[5], math.hypot(src[2] - t[2], src[3] - t[3])))
            need = target[5] + 5
            if src[5] > need:
                angle = math.atan2(target[3] - src[3], target[2] - src[2])
                moves.append([src[0], angle, int(need)])
                continue

        neutrals = [p for p in planets if p[1] == -1]
        if neutrals:
            target = min(neutrals, key=lambda t: math.hypot(src[2] - t[2], src[3] - t[3]))
            if src[5] > target[5] + 1:
                angle = math.atan2(target[3] - src[3], target[2] - src[2])
                moves.append([src[0], angle, max(1, int(src[5] * 0.25))])

    return moves