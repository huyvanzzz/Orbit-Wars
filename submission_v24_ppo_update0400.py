from __future__ import annotations
# Generated from submission_v24_good.py + PPO checkpoint ppo_policy_update_0400.npz.
import dataclasses
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Sequence
import torch
from torch import Tensor
# ==========================================================================
# orbit_lite.constants
# ==========================================================================
"""Capacity and physics constants for the game (must match the engine)."""
# ---------------------------------------------------------------------------
# Capacity — tune based on GPU memory and profiling
# ---------------------------------------------------------------------------
B_DEFAULT: int = 1024   # default games per batch
P_MAX: int = 64         # planet slots per game  (real games have 24-52 planets)
F_MAX: int = 256        # fleet slots per game
A: int = 2              # players per game
# ---------------------------------------------------------------------------
# Physics (must match the game engine)
# ---------------------------------------------------------------------------
BOARD_SIZE: float = 100.0
CENTER: float = 50.0
SUN_RADIUS: float = 10.0
MAX_SHIP_SPEED: float = 6.0
ROT_RADIUS_LIMIT: float = 50.0  # planets with orbital_radius + radius < this orbit
# ---------------------------------------------------------------------------
# Observation — relative ownership encoding
# ---------------------------------------------------------------------------
OWN: int = 0      # slot belongs to the observing player
ENEMY: int = 1    # slot belongs to an opponent
NEUTRAL: int = 2  # slot is unclaimed
DEAD: int = 3     # slot is empty (alive_mask=False)
# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
LIBRARY_K_DEFAULT: int = 100_000  # number of starting states to pre-generate
# ---------------------------------------------------------------------------
# Comets (optional, gated by comets_enabled)
# ---------------------------------------------------------------------------
COMET_EVENTS: int = 5
COMETS_PER_EVENT: int = 4
COMET_PATH_MAX: int = 40
COMET_SPAWN_STEPS: tuple[int, ...] = (50, 150, 250, 350, 450)
COMET_RADIUS: float = 1.0
COMET_PRODUCTION: float = 1.0
# ---------------------------------------------------------------------------
# Early termination — call the game when one player dominates the leaderboard
#
# ---------------------------------------------------------------------------
EARLY_TERM_MARGIN: float = 2.0          # leader_score >= MARGIN * runner_up_score
EARLY_TERM_STREAK_2P: int = 5           # consecutive turns the lead must hold
EARLY_TERM_STREAK_4P: int = 20
EARLY_TERM_PROD_WEIGHT_2P: float = 5.0  # score = 5 * production + 1 * (planet + fleet ships)
EARLY_TERM_SHIP_WEIGHT_2P: float = 1.0
EARLY_TERM_PROD_WEIGHT_4P: float = 1.0  # 4p uses production alone
EARLY_TERM_SHIP_WEIGHT_4P: float = 0.0
# ---------------------------------------------------------------------------
# Episode length (default number of game steps)
# ---------------------------------------------------------------------------
DEFAULT_EPISODE_STEPS: int = 500
# ==========================================================================
# orbit_lite.aiming
# ==========================================================================
"""Orbit-phase helper used by the movement forecaster."""
from torch import Tensor
def orbit_phase_index_from_obs_step(obs_step: Tensor) -> Tensor:
    """Convert the observation ``step`` counter into the engine orbit phase index.
    Orbiting planets update with ``theta = orb_a0 + angvel * g_step`` *before*
    ``g_step`` is incremented for the next observation. The public observation
    carries ``step == g_step`` after that increment, so the implied phase index
    is ``max(0, step - 1)`` (and ``0`` at game start when ``step == 0``).
    """
    s = obs_step.float()
    return (s - (s > 0).to(s.dtype)).clamp(min=0.0)
# ==========================================================================
# orbit_lite.geometry
# ==========================================================================
"""Geometry primitives. Pure tensor functions with no game-state imports."""
import torch
from torch import Tensor
# Pre-compute log(1000) once as a plain Python float for efficiency.
_LOG_1000: float = float(torch.log(torch.tensor(1000.0)).item())
_FLEET_SPEED_LUT_MAX: int = 400
def _fleet_speed_formula(ships: Tensor) -> Tensor:
    """Exact engine-matching speed formula."""
    ratio = (torch.log(ships) / _LOG_1000).clamp(max=1.0)
    return 1.0 + (MAX_SHIP_SPEED - 1.0) * ratio.pow(1.5)
def _build_fleet_speed_lut(max_ships: int) -> Tensor:
    # Index 0 is unused but keeps indexing branch-free for ships >= 1.
    idx = torch.arange(max_ships + 1, dtype=torch.float32).clamp(min=1.0)
    return _fleet_speed_formula(idx)
_FLEET_SPEED_LUT: Tensor = _build_fleet_speed_lut(_FLEET_SPEED_LUT_MAX)
# Per-(device, dtype) cache of the LUT so a CUDA stream isn't synced by an
# H→D copy on every fleet_speed call. Module-level dict, populated lazily.
_FLEET_SPEED_LUT_CACHE: dict[tuple, Tensor] = {}
def _fleet_speed_lut_on(device: torch.device, dtype: torch.dtype) -> Tensor:
    key = (device, dtype)
    cached = _FLEET_SPEED_LUT_CACHE.get(key)
    if cached is None:
        cached = _FLEET_SPEED_LUT.to(device=device, dtype=dtype)
        _FLEET_SPEED_LUT_CACHE[key] = cached
    return cached
# ---------------------------------------------------------------------------
# Pairwise operations  [N] × [M]  →  [N, M]
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Fleet physics
# ---------------------------------------------------------------------------
def fleet_speed(ships: Tensor) -> Tensor:
    """Travel speed for a fleet of ``ships`` ships.
    The engine ship-speed formula::
        speed = 1 + (MAX_SHIP_SPEED - 1) * (log(ships) / log(1000)) ** 1.5
    Args:
        ships: ship count, any shape; values are clamped to ≥ 1.
    Returns:
        speed in ``[1, MAX_SHIP_SPEED]``, same shape as ``ships``.
    """
    s = ships.clamp(min=1.0)
    s_lut = s.clamp(max=float(_FLEET_SPEED_LUT_MAX))
    lo = torch.floor(s_lut).long()
    hi = torch.ceil(s_lut).long()
    frac = s_lut - lo.to(dtype=s.dtype)
    lut = _fleet_speed_lut_on(s.device, s.dtype)
    speed = lut[lo] + (lut[hi] - lut[lo]) * frac
    # Over-range fleets (>``_FLEET_SPEED_LUT_MAX`` ships) use the exact
    # formula. We unconditionally compute it and select via ``torch.where``
    # rather than a ``bool(over.any())`` branch — the latter triggers a
    # host/device sync per call on CUDA which dominated the wall-clock
    # of every kernel that batches fleet_speed inside its inner loop.
    over = s > float(_FLEET_SPEED_LUT_MAX)
    speed_formula = _fleet_speed_formula(s)
    return torch.where(over, speed_formula, speed)
# ---------------------------------------------------------------------------
# Segment–circle intersection (sun / planet collision geometry)
# ---------------------------------------------------------------------------
# ==========================================================================
# orbit_lite.obs
# ==========================================================================
"""Canonical observation parsing into a named :class:`ParsedObs` dataclass.
Converts the raw 7-field observation tensors (produced by
:func:`adapter.single_obs_to_tensor`) into named per-planet/per-fleet fields.
Field index definitions
-----------------------
``planets`` / ``initial_planets``  ``[P, 7]`` float32::
    0 – planet_id   (alive sentinel: id >= 0; padding value: -1)
    1 – owner       (absolute player index; -1 = neutral)
    2 – x           (board coordinates, 0–100)
    3 – y
    4 – radius
    5 – ships       (current count)
    6 – production  (ships added per turn when owned)
``fleets``  ``[F, 7]`` float32::
    0 – fleet_id    (alive sentinel: id >= 0)
    1 – owner
    2 – x
    3 – y
    4 – angle       (radians)
    5 – from_planet_id
    6 – ships
No field indices appear outside this module; all downstream modules consume
:class:`ParsedObs` named fields instead.
"""
from dataclasses import dataclass
import torch
from torch import Tensor
# ---------------------------------------------------------------------------
# ParsedObs
# ---------------------------------------------------------------------------
@dataclass
class ParsedObs:
    """Named per-planet fields decoded from a raw batch observation dict.
    All tensor fields have shape ``[P]`` unless stated otherwise.
    """
    # --- raw planet fields --------------------------------------------------
    alive: Tensor       # bool  – planet_id >= 0 (not a padding slot)
    x: Tensor           # float – current x position (0–100)
    y: Tensor           # float – current y position (0–100)
    r: Tensor           # float – radius
    ships: Tensor       # float – current ship count
    prod: Tensor        # float – production per turn
    owner_abs: Tensor   # float – absolute owner id (-1 = neutral)
    # --- relative ownership masks (computed from owner_abs + player_id) -----
    owned: Tensor       # bool – alive & owner_abs == player_id
    is_enemy: Tensor    # bool – alive & owner_abs >= 0 & owner_abs != player_id
    is_neutral: Tensor  # bool – alive & owner_abs < 0
    # --- orbital parameters (reconstructed from initial_planets) ------------
    orb_r: Tensor       # float – orbital radius; 0.0 for static planets
    orb_a0: Tensor      # float – initial angle from CENTER (radians)
    is_orbiting: Tensor # bool  – True for rotating planets
    # --- game scalars -------------------------------------------------------
    angvel: Tensor      # [B] float – board angular velocity (rad/turn)
    step: Tensor        # [B] float – current game step
    # --- fleet fields -------------------------------------------------------
    #  Available when parse_obs is called with include_fleets=True;
    #  shapes are [F, *] and accessed as attributes rather than being
    #  indexed per-column.
    f_alive: Tensor     # [F] bool
    f_owner: Tensor     # [F] float – absolute owner
    f_x: Tensor         # [F] float
    f_y: Tensor         # [F] float
    f_angle: Tensor     # [F] float – radians
    f_ships: Tensor     # [F] float
    # --- metadata -----------------------------------------------------------
    player_id: int
    P: int
    F: int
    device: torch.device
# ---------------------------------------------------------------------------
# parse_obs
# ---------------------------------------------------------------------------
def parse_obs(obs_tensors: dict, player_id: int | None = None) -> ParsedObs:
    """Decode a raw batch observation dict into a :class:`ParsedObs`.
    Args:
        obs_tensors: dict as produced by ``adapter.single_obs_to_tensor`` or
                     Required keys:
                     ``"planets"`` ``[P, 7]``,
                     ``"initial_planets"`` ``[P, 7]``,
                     ``"fleets"`` ``[F, 7]``,
                     ``"angular_velocity"`` scalar,
                     ``"step"`` scalar,
                     ``"player"`` scalar.
        player_id:   Which player to compute ownership masks for.  Defaults
                     to ``int(obs_tensors["player"][0])``.
    Returns:
        :class:`ParsedObs` with all tensors on the same device as ``planets``.
    """
    planets = obs_tensors["planets"]          # [P, 7]
    initial = obs_tensors["initial_planets"]  # [P, 7]
    fleets = obs_tensors["fleets"]            # [F, 7]
    angvel = obs_tensors["angular_velocity"].float()  # scalar
    step = obs_tensors["step"].float()        # scalar
    if player_id is None:
        player_id = int(obs_tensors["player"].flatten()[0].item())
    P, _ = planets.shape
    F, _ = fleets.shape
    device = planets.device
    # -- planet columns -------------------------------------------------------
    pid = planets[..., 0]        # [P]
    owner_abs = planets[..., 1]
    x = planets[..., 2]
    y = planets[..., 3]
    r = planets[..., 4]
    ships = planets[..., 5]
    prod = planets[..., 6]
    alive = pid >= 0.0
    owned = alive & (owner_abs == float(player_id))
    is_enemy = alive & (owner_abs >= 0.0) & (owner_abs != float(player_id))
    is_neutral = alive & (owner_abs < 0.0)
    # -- orbital parameters from initial_planets ------------------------------
    # A planet is "orbiting" when its distance from the board centre plus its
    # radius is below ROT_RADIUS_LIMIT (mirroring the engine's initialisation
    # logic).  We reconstruct the orbital radius and initial angle from the
    # initial position stored in the observation.
    ix = initial[..., 2]  # [P]
    iy = initial[..., 3]
    i_r = initial[..., 4]  # initial radius (same as current for orbiting)
    dx0 = ix - CENTER
    dy0 = iy - CENTER
    orb_r_raw = torch.sqrt(dx0 * dx0 + dy0 * dy0)
    orb_a0 = torch.atan2(dy0, dx0)
    # Orbiting: alive, initial orbital radius + planet radius < limit, and
    # non-trivially away from the centre (avoids treating dead/padding slots
    # with ix=iy=0 as orbiting).
    is_orbiting = alive & ((orb_r_raw + i_r) < ROT_RADIUS_LIMIT) & (orb_r_raw > 0.5)
    # Static planets carry orb_r = 0 so downstream maths stay correct.
    orb_r = torch.where(is_orbiting, orb_r_raw, torch.zeros_like(orb_r_raw))
    # -- fleet columns --------------------------------------------------------
    f_pid = fleets[..., 0]      # [F]
    f_alive = f_pid >= 0.0
    f_owner = fleets[..., 1]
    f_x = fleets[..., 2]
    f_y = fleets[..., 3]
    f_angle = fleets[..., 4]
    f_ships = fleets[..., 6]
    return ParsedObs(
        alive=alive,
        x=x, y=y, r=r,
        ships=ships, prod=prod,
        owner_abs=owner_abs,
        owned=owned,
        is_enemy=is_enemy,
        is_neutral=is_neutral,
        orb_r=orb_r,
        orb_a0=orb_a0,
        is_orbiting=is_orbiting,
        angvel=angvel,
        step=step,
        f_alive=f_alive,
        f_owner=f_owner,
        f_x=f_x, f_y=f_y,
        f_angle=f_angle,
        f_ships=f_ships,
        player_id=player_id,
        P=P, F=F,
        device=device,
    )
# ==========================================================================
# orbit_lite.movement_aiming
# ==========================================================================
"""Aiming helpers backed by :class:`PlanetMovement`.
This module is intentionally small: it solves angle/ETA for concrete
``(source_slot, target_slot, fleet_size)`` candidates using cached future planet
positions, then masks candidates whose straight path crosses the sun or another
planet.
"""
import torch
from torch import Tensor
LAUNCH_SURFACE_OFFSET: float = 0.1
"""Fleet launch offset from source surface.
Matches Kaggle Orbit Wars engine launch placement:
``start = source + unit(angle) * (source_radius + 0.1)``.
"""
TARGET_HIT_SURFACE_OFFSET: float = 0.0
"""Extra target-surface margin for hit ETA estimation.
Kaggle/local engines register fleet-vs-planet contact when the fleet center
enters the target planet radius. ``0.0`` keeps ETA aligned with that rule.
"""
KAGGLE_SUN_RADIUS: float = SUN_RADIUS
"""Sun collision radius used by Kaggle/local engines."""
def _swept_pair_hit_mask(
    ax: Tensor,
    ay: Tensor,
    bx: Tensor,
    by: Tensor,
    p0x: Tensor,
    p0y: Tensor,
    p1x: Tensor,
    p1y: Tensor,
    r: Tensor,
) -> Tensor:
    d0x = ax - p0x
    d0y = ay - p0y
    dvx = (bx - ax) - (p1x - p0x)
    dvy = (by - ay) - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    near_static = a < 1e-12
    c_hit = c <= 0.0
    disc = b * b - 4.0 * a * c
    has_root = disc >= 0.0
    safe_a = torch.where(near_static, torch.ones_like(a), a)
    sq = torch.sqrt(torch.clamp(disc, min=0.0))
    t1 = (-b - sq) / (2.0 * safe_a)
    t2 = (-b + sq) / (2.0 * safe_a)
    quad_hit = has_root & (t2 >= 0.0) & (t1 <= 1.0)
    return torch.where(near_static, c_hit, quad_hit)
# ==========================================================================
# orbit_lite.movement
# ==========================================================================
"""Future planet/comet movement cache + garrison projection for one game.
``PlanetMovement`` predicts planet and comet positions from an observation, keeps
a short rolling horizon, tracks in-flight fleets, and projects per-planet owner /
ships over the horizon (the do-nothing garrison forecast agents plan against).
"""
from dataclasses import dataclass
import torch
from torch import Tensor
DEFAULT_MOVEMENT_HORIZON = 20
DEFAULT_DRIFT_EPSILON = 1e-4
DEFAULT_MAX_TRACKED_FLEETS = 64
@dataclass(frozen=True)
class MovementConfig:
    """Configuration for ``PlanetMovement`` construction and updates."""
    movement_horizon: int = DEFAULT_MOVEMENT_HORIZON
    drift_epsilon: float = DEFAULT_DRIFT_EPSILON
    track_fleets: bool = False
    player_count: int | None = None
    max_tracked_fleets: int = DEFAULT_MAX_TRACKED_FLEETS
@dataclass(frozen=True)
class PlanetGarrisonStatus:
    """Projected planet ownership and garrison ships over cached future steps.
    ``owner`` / ``ships`` are *post-combat* values: what the planet looks like at
    the end of each future step assuming the agent does **not** act. They are the
    right oracle for "what will be there in N turns if I do nothing."
    ``pre_combat_owner`` / ``pre_combat_ships`` are the planet state *just
    before* combat resolution at each future step — after that step's production
    has been credited but before any same-step arrivals are applied. Agents
    planning their own arrival on step ``k`` should consult these (plus the
    per-step ``arrivals_by_owner``) and apply the engine combat rule themselves:
    treating their own send as an additional same-step attacker. They are
    populated only when fleet tracking is enabled.
    ``arrivals_by_owner`` mirrors ``PlanetMovement.fleet_buckets`` at the
    requested planet slots: per-step per-owner ship totals arriving on a given
    target. Shape ``[*prefix, H, A]`` where ``A`` is the number of agents. ``None``
    when fleet tracking is off.
    """
    owner: Tensor
    ships: Tensor
    pre_combat_owner: Tensor | None = None
    pre_combat_ships: Tensor | None = None
    arrivals_by_owner: Tensor | None = None
@dataclass
class PlanetMovement:
    """Rolling cache of future planet positions for a single game.
    Tensor shapes:
    - ``x``, ``y``, ``alive_by_step``: ``[H + 1, P]``
    - ``planet_ids``, ``radii``: ``[P]``
    - ``base_step``: scalar
    - optional ``fleet_buckets``: ``[P, H, A]``
    ``k == 0`` is the observation frame used to build the cache, and ``k`` is
    the number of future movement steps from that frame.
    """
    x: Tensor
    y: Tensor
    alive_by_step: Tensor
    planet_ids: Tensor
    radii: Tensor
    planet_owner: Tensor
    planet_ships: Tensor
    planet_prod: Tensor
    base_step: Tensor
    comet_planet_ids: Tensor
    comet_path_index: Tensor
    movement_horizon: int = DEFAULT_MOVEMENT_HORIZON
    drift_epsilon: float = DEFAULT_DRIFT_EPSILON
    track_fleets: bool = False
    player_count: int | None = None
    max_tracked_fleets: int = DEFAULT_MAX_TRACKED_FLEETS
    fleet_buckets: Tensor | None = None
    fleet_last_step: Tensor | None = None
    tracked_fleet_ids: Tensor | None = None
    tracked_fleet_eta: Tensor | None = None
    tracked_fleet_target_slot: Tensor | None = None
    # Per-entry owner / ship-count of the recorded arrival. Required so
    # ``_reconcile_obs_fleets`` can subtract a phantom's contribution from
    # ``fleet_buckets`` when its fleet id vanishes from obs.
    tracked_fleet_owner: Tensor | None = None
    tracked_fleet_ships: Tensor | None = None
    garrison_owner_cache: Tensor | None = None
    garrison_ships_cache: Tensor | None = None
    garrison_pre_combat_owner_cache: Tensor | None = None
    garrison_pre_combat_ships_cache: Tensor | None = None
    garrison_dirty_from: Tensor | None = None
    # Per-batch pending launches awaiting fleet-id reconciliation against the
    # next observation. Each lane carries up to ``pending_*`` columns of
    # stashed-launch metadata; empty slots are marked by ``pending_owners ==
    # -1``. See ``stash_pending_own_launches`` and
    # ``_reconcile_pending_own_launches``. ``next_fleet_id`` and the step at
    # stash time are stored per-entry so multi-owner stash within one turn
    # works.
    pending_source_planets: Tensor | None = None   # [L] long  (-1 = empty)
    pending_ships: Tensor | None = None            # [L] long
    pending_angle: Tensor | None = None            # [L] dtype
    pending_target_slots: Tensor | None = None     # [L] long
    pending_eta: Tensor | None = None              # [L] dtype
    pending_owners: Tensor | None = None           # [L] long  (-1 = empty)
    pending_prev_nfid: Tensor | None = None        # [L] long
    pending_stash_step: Tensor | None = None       # [L] long
    @property
    def P(self) -> int:
        return int(self.planet_ids.shape[0])
    @property
    def device(self) -> torch.device:
        return self.x.device
    @property
    def dtype(self) -> torch.dtype:
        return self.x.dtype
    @property
    def config(self) -> MovementConfig:
        """Return the explicit movement config used by this cache."""
        return MovementConfig(
            movement_horizon=int(self.movement_horizon),
            drift_epsilon=float(self.drift_epsilon),
            track_fleets=bool(self.track_fleets),
            player_count=self.player_count,
            max_tracked_fleets=int(self.max_tracked_fleets),
        )
    @classmethod
    def from_obs_tensors(
        cls,
        obs_tensors: dict,
        *,
        config: MovementConfig | None = None,
        movement_horizon: int = DEFAULT_MOVEMENT_HORIZON,
        drift_epsilon: float = DEFAULT_DRIFT_EPSILON,
        track_fleets: bool = False,
        player_count: int | None = None,
        max_tracked_fleets: int = DEFAULT_MAX_TRACKED_FLEETS,
    ) -> "PlanetMovement":
        """Build a fresh movement cache from batched observation tensors.
        The cache has movement parameters plus optional fleet tracking:
        - ``movement_horizon``: number of future steps cached.
        - ``drift_epsilon``: tolerated positional drift before rebuild.
        - ``track_fleets``: opt-in arrival buckets shaped ``[P, H, A]``.
        - ``player_count``: known player count (2 or 4), or inferred at turn 0.
        - ``max_tracked_fleets``: capacity per batch lane for in-flight fleet-id ledger rows.
        """
        cfg = config if config is not None else MovementConfig(
            movement_horizon=int(movement_horizon),
            drift_epsilon=float(drift_epsilon),
            track_fleets=bool(track_fleets),
            player_count=player_count,
            max_tracked_fleets=int(max_tracked_fleets),
        )
        built = _build_future_from_obs(obs_tensors, int(cfg.movement_horizon))
        resolved_player_count = _resolve_player_count(obs_tensors, cfg.player_count) if cfg.track_fleets else cfg.player_count
        movement = cls(
            x=built["x"],
            y=built["y"],
            alive_by_step=built["alive_by_step"],
            planet_ids=built["planet_ids"],
            radii=built["radii"],
            planet_owner=built["owner"],
            planet_ships=built["ships"],
            planet_prod=built["prod"],
            base_step=built["step"],
            comet_planet_ids=built["comet_planet_ids"],
            comet_path_index=built["comet_path_index"],
            movement_horizon=int(cfg.movement_horizon),
            drift_epsilon=float(cfg.drift_epsilon),
            track_fleets=bool(cfg.track_fleets),
            player_count=resolved_player_count,
            max_tracked_fleets=int(cfg.max_tracked_fleets),
        )
        if movement.track_fleets:
            movement._init_fleet_tracking(obs_tensors, reset_ledger=True)
            movement._ingest_obs_fleets(obs_tensors)
        return movement
    def update(self, obs_tensors: dict) -> "PlanetMovement":
        """Refresh this cache for a new observation (single game).
        If the current observation matches the cached prediction the trajectory
        is kept (same step) or rolled forward by one step. Numeric drift, step
        jumps, shape/device changes, or planet/comet identity changes trigger a
        full rebuild from the new observation.
        """
        planets = obs_tensors["planets"]
        if (
            planets.device != self.device
            or planets.shape[0] != self.P
            or int(self.x.shape[0]) != int(self.movement_horizon) + 1
        ):
            fresh = type(self).from_obs_tensors(
                obs_tensors,
                movement_horizon=self.movement_horizon,
                drift_epsilon=self.drift_epsilon,
                track_fleets=self.track_fleets,
                player_count=self.player_count,
                max_tracked_fleets=int(self.max_tracked_fleets),
            )
            self._copy_from(fresh)
            return self
        if self.track_fleets:
            current_player_count = _resolve_player_count(obs_tensors, self.player_count)
            if (
                self.fleet_buckets is None
                or self.fleet_last_step is None
                or self.tracked_fleet_ids is None
                or tuple(self.fleet_buckets.shape) != (
                    self.P,
                    int(self.movement_horizon),
                    int(current_player_count),
                )
                or self.fleet_buckets.device != self.device
                or int(self.tracked_fleet_ids.shape[0]) < int(self.max_tracked_fleets)
            ):
                self.player_count = int(current_player_count)
                self._init_fleet_tracking(obs_tensors, reset_ledger=True)
        obs_for_decision = parse_obs(obs_tensors)
        H = int(self.movement_horizon)
        planet_ids_now = planets[..., 0].long()
        radii_now = planets[..., 4].to(dtype=self.dtype)
        owner_now = planets[..., 1].to(device=self.device, dtype=torch.long)
        owner_now = torch.where(
            obs_for_decision.alive, owner_now, torch.full_like(owner_now, -1)
        )
        ships_now = planets[..., 5].to(device=self.device, dtype=self.dtype)
        prod_now = planets[..., 6].to(device=self.device, dtype=self.dtype)
        step_now = obs_for_decision.step.to(device=self.device, dtype=torch.long)
        comet_ids_now, comet_idx_now = _comet_metadata(obs_tensors, self.device)
        current_obs_x = planets[..., 2].to(device=self.device, dtype=self.dtype)
        current_obs_y = planets[..., 3].to(device=self.device, dtype=self.dtype)
        current_alive = obs_for_decision.alive
        ids_same = bool((planet_ids_now == self.planet_ids).all())
        same_step = bool(step_now == self.base_step)
        next_step = bool(step_now == (self.base_step + 1))
        comet_same = _same_2d(comet_ids_now, self.comet_planet_ids)
        comet_idx_same = _same_2d(comet_idx_now, self.comet_path_index)
        expected_next_idx = torch.where(
            self.comet_path_index >= 0,
            self.comet_path_index + 1,
            self.comet_path_index,
        )
        comet_idx_next = _same_2d(comet_idx_now, expected_next_idx)
        same_alive_ok = bool((current_alive == self.alive_by_step[0]).all())
        next_alive_ok = bool((current_alive == self.alive_by_step[1]).all())
        same_drift_ok = _position_matches(
            self.x[0], self.y[0], current_obs_x, current_obs_y,
            current_alive, float(self.drift_epsilon),
        )
        next_drift_ok = _position_matches(
            self.x[1], self.y[1], current_obs_x, current_obs_y,
            current_alive, float(self.drift_epsilon),
        )
        keep = ids_same and same_step and comet_same and comet_idx_same and same_alive_ok and same_drift_ok
        roll = ids_same and next_step and comet_same and comet_idx_next and next_alive_ok and next_drift_ok
        rebuild = not (keep or roll)
        if rebuild:
            built = _build_future_from_obs(obs_tensors, H)
        elif roll:
            # Roll-only path: build just the new last frame at offset H.
            last_offset = torch.tensor([H], dtype=torch.long, device=self.device)
            built = _build_future_from_obs(obs_tensors, H, offsets=last_offset)
        else:
            built = None
        if roll:
            assert built is not None
            self.x[:-1] = self.x[1:].clone()
            self.y[:-1] = self.y[1:].clone()
            self.alive_by_step[:-1] = self.alive_by_step[1:].clone()
            self.x[-1] = built["x"][-1]
            self.y[-1] = built["y"][-1]
            self.alive_by_step[-1] = built["alive_by_step"][-1]
            self._roll_garrison_projection()
        if rebuild:
            assert built is not None
            self.x[:] = built["x"]
            self.y[:] = built["y"]
            self.alive_by_step[:] = built["alive_by_step"]
            self._mark_garrison_dirty_all(0)
        if roll or rebuild:
            self.planet_ids[:] = planet_ids_now
            self.radii[:] = radii_now
            self.base_step = step_now
            self.comet_planet_ids = comet_ids_now
            self.comet_path_index = comet_idx_now
        self._refresh_garrison_base({
            "planet_ids": planet_ids_now,
            "radii": radii_now,
            "owner": owner_now,
            "ships": ships_now,
            "prod": prod_now,
            "step": step_now,
        })
        if self.track_fleets:
            self._roll_fleet_buckets_phase1(step_now)
            if rebuild and not ids_same:
                self._reset_fleet_tracking()
            self._reconcile_pending_own_launches(obs_tensors)
            self._ingest_obs_fleets(obs_tensors)
            self._reconcile_obs_fleets(obs_tensors)
        return self
    def all_positions(self, k: int) -> tuple[Tensor, Tensor]:
        """Return all planet positions ``k`` steps ahead as ``[P]``."""
        idx = self._k_index(k)
        return self.x[idx], self.y[idx]
    def alive_at(self, k: int) -> Tensor:
        """Return alive mask ``k`` steps ahead as ``[P]``."""
        return self.alive_by_step[self._k_index(k)]
    def position_at_slots(self, slots: Tensor, k: int) -> tuple[Tensor, Tensor]:
        """Gather future positions for slot indices of any shape."""
        slots = slots.to(device=self.device, dtype=torch.long).clamp(0, max(self.P - 1, 0))
        px, py = self.all_positions(k)
        out_x = px[slots].to(dtype=self.dtype)
        out_y = py[slots].to(dtype=self.dtype)
        return out_x, out_y
    def pairwise_distance(self, k: int) -> Tensor:
        """Return all pairwise planet distances ``k`` steps ahead, ``[P, P]``."""
        px, py = self.all_positions(k)
        dx = px.unsqueeze(1) - px.unsqueeze(0)
        dy = py.unsqueeze(1) - py.unsqueeze(0)
        return torch.sqrt((dx * dx + dy * dy).clamp(min=0.0))
    def garrison_status(self, planet_slots: Tensor | None = None, *, max_horizon: int | None = None) -> PlanetGarrisonStatus:
        """Return projected owner and ships for selected planet slots.
        The output time axis is ``H + 1``: ``k=0`` is the current observation,
        and ``k=1..H`` are post-production/post-combat states for future turns.
        Fleet tracking must be enabled so arrivals are available.
        """
        self._require_fleet_buckets()
        slots, out_prefix = self._normalize_garrison_slots(planet_slots)
        requested_horizon = int(
            self.movement_horizon if max_horizon is None else max(0, min(int(max_horizon), int(self.movement_horizon)))
        )
        self._refresh_garrison_projection(slots, requested_horizon=requested_horizon)
        assert self.garrison_owner_cache is not None
        assert self.garrison_ships_cache is not None
        assert self.garrison_dirty_from is not None
        owner = self.garrison_owner_cache[slots][:, : requested_horizon + 1].reshape(*out_prefix, requested_horizon + 1)
        ships = self.garrison_ships_cache[slots][:, : requested_horizon + 1].reshape(*out_prefix, requested_horizon + 1)
        pre_combat_owner: Tensor | None = None
        pre_combat_ships: Tensor | None = None
        if (
            self.garrison_pre_combat_owner_cache is not None
            and self.garrison_pre_combat_ships_cache is not None
        ):
            pre_combat_owner = (
                self.garrison_pre_combat_owner_cache[slots][:, : requested_horizon + 1]
                .reshape(*out_prefix, requested_horizon + 1)
            )
            pre_combat_ships = (
                self.garrison_pre_combat_ships_cache[slots][:, : requested_horizon + 1]
                .reshape(*out_prefix, requested_horizon + 1)
            )
        arrivals_by_owner: Tensor | None = None
        if self.fleet_buckets is not None and requested_horizon > 0:
            # ``fleet_buckets`` shape: [P, H, A]. Select the slots to produce
            # [*out_prefix, requested_horizon, A]; then left-pad a zero step-0
            # frame so the time axis lines up with the owner/ships caches (which
            # have an extra ``k=0`` observation slot).
            A = int(self.fleet_buckets.shape[-1])
            arrivals_full = (
                self.fleet_buckets[slots]
                .reshape(*out_prefix, int(self.movement_horizon), A)
            )
            # Trim/pad to the requested horizon: k=0 has no arrivals; k=1..H map
            # to fleet_buckets[..., 0..H-1, :].
            arrivals_trimmed = arrivals_full[..., :requested_horizon, :]
            zero_frame = torch.zeros(
                *out_prefix, 1, A, dtype=arrivals_trimmed.dtype, device=self.device
            )
            arrivals_by_owner = torch.cat([zero_frame, arrivals_trimmed], dim=-2)
        status = PlanetGarrisonStatus(
            owner=owner,
            ships=ships,
            pre_combat_owner=pre_combat_owner,
            pre_combat_ships=pre_combat_ships,
            arrivals_by_owner=arrivals_by_owner,
        )
        return status
    def _clear_pending_mask(self, mask: Tensor) -> None:
        """Reset pending-launch slots selected by ``mask`` (``[L]`` bool)."""
        if self.pending_owners is None:
            return
        self.pending_owners[mask] = -1
        assert self.pending_source_planets is not None
        self.pending_source_planets[mask] = -1
        assert self.pending_ships is not None
        self.pending_ships[mask] = 0
        assert self.pending_angle is not None
        self.pending_angle[mask] = 0.0
        assert self.pending_target_slots is not None
        self.pending_target_slots[mask] = -1
        assert self.pending_eta is not None
        self.pending_eta[mask] = 0.0
        assert self.pending_prev_nfid is not None
        self.pending_prev_nfid[mask] = 0
        assert self.pending_stash_step is not None
        self.pending_stash_step[mask] = -1
    def _ensure_pending_capacity(self, needed: int) -> None:
        """Ensure ``pending_*`` tensors have at least ``needed`` empty slots."""
        device = self.device
        if self.pending_owners is None:
            initial = max(4, int(needed))
            shape = (initial,)
            self.pending_owners = torch.full(shape, -1, dtype=torch.long, device=device)
            self.pending_source_planets = torch.full(shape, -1, dtype=torch.long, device=device)
            self.pending_ships = torch.zeros(shape, dtype=torch.long, device=device)
            self.pending_angle = torch.zeros(shape, dtype=self.dtype, device=device)
            self.pending_target_slots = torch.full(shape, -1, dtype=torch.long, device=device)
            self.pending_eta = torch.zeros(shape, dtype=self.dtype, device=device)
            self.pending_prev_nfid = torch.zeros(shape, dtype=torch.long, device=device)
            self.pending_stash_step = torch.full(shape, -1, dtype=torch.long, device=device)
            return
        assert self.pending_owners is not None
        empty_count = int((self.pending_owners == -1).sum().item())
        shortage = int(needed) - empty_count
        if shortage <= 0:
            return
        cur_L = int(self.pending_owners.shape[0])
        # Grow generously to amortize.
        extra = max(shortage, cur_L)
        new_L = cur_L + extra
        def _grow(t: Tensor, fill: float | int) -> Tensor:
            extension = torch.full((new_L - cur_L,), fill, dtype=t.dtype, device=device)
            return torch.cat([t, extension], dim=0)
        self.pending_owners = _grow(self.pending_owners, -1)
        assert self.pending_source_planets is not None
        self.pending_source_planets = _grow(self.pending_source_planets, -1)
        assert self.pending_ships is not None
        self.pending_ships = _grow(self.pending_ships, 0)
        assert self.pending_angle is not None
        self.pending_angle = _grow(self.pending_angle, 0.0)
        assert self.pending_target_slots is not None
        self.pending_target_slots = _grow(self.pending_target_slots, -1)
        assert self.pending_eta is not None
        self.pending_eta = _grow(self.pending_eta, 0.0)
        assert self.pending_prev_nfid is not None
        self.pending_prev_nfid = _grow(self.pending_prev_nfid, 0)
        assert self.pending_stash_step is not None
        self.pending_stash_step = _grow(self.pending_stash_step, -1)
    def stash_pending_own_launches(
        self,
        *,
        owner_id: int | Tensor,
        source_slots: Tensor,
        ships: Tensor,
        angle: Tensor,
        target_slots: Tensor,
        eta: Tensor,
        valid: Tensor,
        prev_next_fleet_id: int | Tensor,
    ) -> None:
        """Stash this turn\'s own launches for ID reconciliation on the next obs.
        The caller has already added the bucket contribution via
        :meth:`record_fleet_arrivals` (with ``fleet_ids=None``) but must not
        seed the ``tracked_fleet_ids`` ledger yet — the engine assigns IDs in
        slot-major order across all players, so the agent cannot know its
        real IDs at action time. We stash ``(source_planet_id, ships, angle)``
        for each valid launch in emission order; the next call to
        :meth:`update` pairs them against ``obs.fleets`` entries with
        ``id >= prev_next_fleet_id`` and ``owner == owner_id`` (which are the
        engine's actual IDs for this slot's launches this turn) and writes
        the ledger with those real IDs.
        ``prev_next_fleet_id`` is ``obs.next_fleet_id`` at action time (scalar).
        Inputs are ``[L_in]`` (or broadcastable). Pending rows are appended into
        free slots, growing capacity as needed.
        """
        if not self.track_fleets:
            return
        device = self.device
        valid_mask = valid.to(device=device, dtype=torch.bool).reshape(-1)     # [L_in]
        if not bool(valid_mask.any()):
            return
        src = source_slots.to(device=device, dtype=torch.long).reshape(-1)
        ships_t = ships.to(device=device, dtype=torch.long).reshape(-1)
        angle_t = angle.to(device=device, dtype=self.dtype).reshape(-1)
        tgt_t = target_slots.to(device=device, dtype=torch.long).reshape(-1)
        eta_t = eta.to(device=device, dtype=self.dtype).reshape(-1)
        # Resolve source slot -> planet_id.
        src_safe = src.clamp(min=0, max=max(int(self.P) - 1, 0))
        source_planet_ids = self.planet_ids[src_safe]                          # [L_in]
        L_in = int(valid_mask.shape[0])
        if isinstance(prev_next_fleet_id, Tensor):
            prev_nfid_scalar = int(prev_next_fleet_id.flatten()[0].item())
        else:
            prev_nfid_scalar = int(prev_next_fleet_id)
        prev_nfid_L = torch.full((L_in,), prev_nfid_scalar, dtype=torch.long, device=device)
        owner_scalar = int(owner_id.flatten()[0].item()) if isinstance(owner_id, Tensor) else int(owner_id)
        owner_L = torch.full((L_in,), owner_scalar, dtype=torch.long, device=device)
        stash_step_scalar = int(self.base_step.item()) if isinstance(self.base_step, Tensor) else -1
        stash_step_L = torch.full((L_in,), stash_step_scalar, dtype=torch.long, device=device)
        # Clear any prior pending entries for this owner — a repeat stash for the
        # same owner within a turn replaces the previous stash.
        if self.pending_owners is not None:
            same_owner = self.pending_owners == owner_scalar                   # [L]
            if bool(same_owner.any()):
                self._clear_pending_mask(same_owner)
        per_needed = int(valid_mask.sum().item())
        self._ensure_pending_capacity(per_needed)
        assert self.pending_owners is not None
        # Place valid inputs (in ascending order) into the first empty pending
        # slots (ascending) — preserving emission order.
        empty_slots = torch.nonzero(self.pending_owners == -1, as_tuple=True)[0]
        k_in = torch.nonzero(valid_mask, as_tuple=True)[0]                     # [N]
        slot_in_pending = empty_slots[: k_in.numel()]                          # [N]
        self.pending_owners[slot_in_pending] = owner_L[k_in]
        assert self.pending_source_planets is not None
        self.pending_source_planets[slot_in_pending] = source_planet_ids[k_in]
        assert self.pending_ships is not None
        self.pending_ships[slot_in_pending] = ships_t[k_in]
        assert self.pending_angle is not None
        self.pending_angle[slot_in_pending] = angle_t[k_in]
        assert self.pending_target_slots is not None
        self.pending_target_slots[slot_in_pending] = tgt_t[k_in]
        assert self.pending_eta is not None
        self.pending_eta[slot_in_pending] = eta_t[k_in]
        assert self.pending_prev_nfid is not None
        self.pending_prev_nfid[slot_in_pending] = prev_nfid_L[k_in]
        assert self.pending_stash_step is not None
        self.pending_stash_step[slot_in_pending] = stash_step_L[k_in]
    def _reconcile_pending_own_launches(self, obs_tensors: dict) -> None:
        """Pair stashed launches against obs.fleets and seed the ledger with
        engine-assigned IDs.
        Matched stash entries (same owner / source / ships / angle, id >=
        prev_nfid) seed the ledger with their real fleet IDs. Unmatched
        entries are treated as vanished mid-flight — the engine can destroy
        a freshly-launched fleet on its first move via an obstacle the
        agent's swept-pair didn't predict (most commonly a comet between
        source and the predicted target) — and we undo the bucket-arrival
        contribution recorded at stash time so garrison projections stay
        consistent. Still hard-fails when two pending entries match the same
        obs fleet, which signals identical multi-launch from the same source
        that the engine processed in an unexpected order. Extra obs fleets
        (engine-created launches that the planner's swept-pair couldn't
        track, e.g., launches headed OOB) are left alone for
        ``_ingest_obs_fleets`` to handle.
        """
        if not self.track_fleets:
            return
        if self.pending_owners is None or self.tracked_fleet_ids is None:
            return
        active_mask = self.pending_owners != -1                                # [L]
        if not bool(active_mask.any()):
            return
        device = self.device
        step_tensor = obs_tensors.get("step")
        if step_tensor is not None:
            assert self.pending_stash_step is not None
            step_scalar = int(step_tensor.flatten()[0].item()) if isinstance(step_tensor, Tensor) else int(step_tensor)
            advanced = step_scalar > self.pending_stash_step                   # [L]
            active_mask = active_mask & advanced
        if not bool(active_mask.any()):
            return
        fleets = obs_tensors["fleets"].to(device=device)                       # [F, 7]
        fleet_ids = fleets[..., 0].to(dtype=torch.long)                        # [F]
        obs_owner = fleets[..., 1].to(dtype=torch.long)                        # [F]
        obs_angle = fleets[..., 4].to(dtype=self.dtype)                        # [F]
        obs_from = fleets[..., 5].to(dtype=torch.long)                         # [F]
        obs_ships = fleets[..., 6].to(dtype=torch.long)                        # [F]
        assert self.pending_owners is not None
        assert self.pending_source_planets is not None
        assert self.pending_ships is not None
        assert self.pending_angle is not None
        assert self.pending_target_slots is not None
        assert self.pending_eta is not None
        assert self.pending_prev_nfid is not None
        # Pairwise match every obs fleet (rows) against every active pending
        # entry (cols) -> [F, L].
        match_FL = (
            active_mask.unsqueeze(0)
            & (fleet_ids.unsqueeze(1) >= 0)
            & (obs_owner.unsqueeze(1) == self.pending_owners.unsqueeze(0))
            & (obs_from.unsqueeze(1) == self.pending_source_planets.unsqueeze(0))
            & (obs_ships.unsqueeze(1) == self.pending_ships.unsqueeze(0))
            & (obs_angle.unsqueeze(1) == self.pending_angle.unsqueeze(0))
            & (fleet_ids.unsqueeze(1) >= self.pending_prev_nfid.unsqueeze(0))
        )  # [F, L]
        # For each active pending entry, pick the smallest matching obs id.
        INF = torch.iinfo(torch.long).max
        id_for_match = torch.where(
            match_FL,
            fleet_ids.unsqueeze(1).expand_as(match_FL),
            torch.full_like(match_FL, INF, dtype=torch.long),
        )                                                                      # [F, L]
        chosen_id, _ = id_for_match.min(dim=0)                                 # [L]
        # eta_remaining = ceil(stash.eta) - 1; one turn has passed. ``eta_now
        # <= 0`` means the fleet arrived this turn (resolved + removed from obs),
        # so we don't expect an obs match. For eta_now > 0 a missing match means
        # the engine destroyed the fleet mid-flight; treat as vanished: drop the
        # pending entry, skip the ledger insert, and undo the pre-recorded bucket
        # arrival so garrison projections aren't biased by a phantom.
        eta_now = torch.ceil(self.pending_eta).to(dtype=torch.long) - 1
        expect_obs_match = active_mask & (eta_now > 0)
        no_match = expect_obs_match & (chosen_id == INF)
        matched = expect_obs_match & (chosen_id != INF)
        # Detect duplicate assignments among matched entries: two pending entries
        # pointing at the same chosen_id (identical multi-launch from one source
        # processed in an unexpected order).
        if int(active_mask.shape[0]) > 1:
            chosen_for_matched = torch.where(
                matched, chosen_id, torch.full_like(chosen_id, INF)
            )
            sorted_ids, _ = chosen_for_matched.sort()
            dup = bool(
                ((sorted_ids[1:] == sorted_ids[:-1]) & (sorted_ids[1:] != INF)).any()
            )
            if dup:
                raise AssertionError(
                    "Pending-launch reconciliation: multiple pending entries "
                    "resolved to the same engine fleet id. This usually means "
                    "multi-launch from the same source with identical "
                    "(ships, angle) tuples processed in an unexpected order."
                )
        if bool(matched.any()):
            l_idx = torch.where(matched)[0]
            real_ids = chosen_id[l_idx]
            self._ledger_bulk_insert(
                real_ids,
                eta_now[l_idx],
                self.pending_target_slots[l_idx],
                self.pending_owners[l_idx],
                self.pending_ships[l_idx].to(dtype=self.dtype),
            )
        if bool(no_match.any()):
            self._decrement_unmatched_arrivals(no_match)
        # Clear ALL pending entries we just reconciled (eta<=0 cases never make
        # it to the ledger but shouldn't linger either).
        self._clear_pending_mask(active_mask)
    def _decrement_unmatched_arrivals(self, no_match: Tensor) -> None:
        """Undo the bucket-arrival contribution recorded for a launch that
        vanished before reaching its predicted target.
        The pre-record sat at ``buckets[target_slot, ceil(eta)-1, owner]`` at
        stash time. By the time this runs, ``_roll_fleet_buckets_phase1`` has
        already shifted the bucket one step forward, so the relevant index is
        ``ceil(eta)-2 == eta_now-1``. Entries that already rolled off the
        horizon leave nothing to decrement and are skipped.
        """
        assert self.pending_eta is not None
        assert self.pending_owners is not None
        assert self.pending_ships is not None
        assert self.pending_target_slots is not None
        buckets = self._require_fleet_buckets()
        eta_now = torch.ceil(self.pending_eta).to(dtype=torch.long) - 1
        h_idx_now = eta_now - 1
        H = int(self.movement_horizon)
        Aowner = int(buckets.shape[2])
        valid = (
            no_match
            & (h_idx_now >= 0)
            & (h_idx_now < H)
            & (self.pending_target_slots >= 0)
            & (self.pending_target_slots < int(self.P))
            & (self.pending_owners >= 0)
            & (self.pending_owners < Aowner)
            & (self.pending_ships > 0)
        )
        if not bool(valid.any()):
            return
        target = self.pending_target_slots[valid]
        h_idx_sel = h_idx_now[valid]
        owner_sel = self.pending_owners[valid]
        ships_sel = self.pending_ships[valid].to(dtype=self.dtype)
        buckets.index_put_(
            (target, h_idx_sel, owner_sel),
            -ships_sel,
            accumulate=True,
        )
        self._mark_garrison_dirty(target, h_idx_sel + 1)
    def record_fleet_arrivals(
        self,
        *,
        target_slots: Tensor,
        owner_ids: Tensor | int,
        ships: Tensor,
        eta: Tensor,
        valid: Tensor | None = None,
    ) -> None:
        """Add predicted arrivals into the fleet buckets.
        ``eta`` is expressed in steps from the current observation frame; bucket
        ``eta=1`` is stored at horizon index ``0``.
        """
        buckets = self._require_fleet_buckets()
        target_slots, ships, eta = torch.broadcast_tensors(
            target_slots.to(device=self.device, dtype=torch.long),
            ships.to(device=self.device, dtype=self.dtype),
            eta.to(device=self.device, dtype=self.dtype),
        )
        if isinstance(owner_ids, int):
            owner = torch.full_like(target_slots, int(owner_ids), dtype=torch.long, device=self.device)
        else:
            owner = torch.broadcast_to(owner_ids.to(device=self.device, dtype=torch.long), target_slots.shape)
        if valid is None:
            valid_mask = torch.ones_like(target_slots, dtype=torch.bool)
        else:
            valid_mask = torch.broadcast_to(valid.to(device=self.device, dtype=torch.bool), target_slots.shape)
        h_idx = torch.ceil(eta).to(dtype=torch.long) - 1
        valid_mask = (
            valid_mask
            & (target_slots >= 0)
            & (target_slots < self.P)
            & (owner >= 0)
            & (owner < int(buckets.shape[2]))
            & (h_idx >= 0)
            & (h_idx < int(self.movement_horizon))
            & (ships > 0.0)
        )
        if not bool(valid_mask.any()):
            return
        buckets.index_put_(
            (
                target_slots[valid_mask],
                h_idx[valid_mask],
                owner[valid_mask],
            ),
            ships[valid_mask],
            accumulate=True,
        )
        self._mark_garrison_dirty(
            target_slots[valid_mask],
            h_idx[valid_mask] + 1,
        )
    def _normalize_garrison_slots(self, planet_slots: Tensor | None) -> tuple[Tensor, torch.Size]:
        if planet_slots is None:
            slots = torch.arange(self.P, dtype=torch.long, device=self.device)
            return slots, slots.shape
        raw = planet_slots.to(device=self.device, dtype=torch.long)
        out_prefix = raw.shape
        slots = raw.reshape(-1).clamp(0, max(self.P - 1, 0))
        return slots, out_prefix
    def _ensure_garrison_cache(self) -> None:
        self._ensure_garrison_cache_impl()
    def _ensure_garrison_cache_impl(self) -> None:
        expected_owner = (self.P, int(self.movement_horizon) + 1)
        expected_dirty = (self.P,)
        if (
            self.garrison_owner_cache is not None
            and self.garrison_ships_cache is not None
            and self.garrison_pre_combat_owner_cache is not None
            and self.garrison_pre_combat_ships_cache is not None
            and self.garrison_dirty_from is not None
            and tuple(self.garrison_owner_cache.shape) == expected_owner
            and tuple(self.garrison_ships_cache.shape) == expected_owner
            and tuple(self.garrison_pre_combat_owner_cache.shape) == expected_owner
            and tuple(self.garrison_pre_combat_ships_cache.shape) == expected_owner
            and tuple(self.garrison_dirty_from.shape) == expected_dirty
            and self.garrison_owner_cache.device == self.device
            and self.garrison_ships_cache.device == self.device
        ):
            return
        horizon = int(self.movement_horizon)
        self.garrison_owner_cache = torch.full(
            (self.P, horizon + 1),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self.garrison_ships_cache = torch.zeros(
            self.P,
            horizon + 1,
            dtype=self.dtype,
            device=self.device,
        )
        # Pre-combat caches: planet state just before each step's combat (after
        # production has been credited). At k=0 there is no prior step, so the
        # observation IS both pre- and post-combat.
        self.garrison_pre_combat_owner_cache = self.garrison_owner_cache.clone()
        self.garrison_pre_combat_ships_cache = self.garrison_ships_cache.clone()
        self.garrison_owner_cache[:, 0] = self.planet_owner
        self.garrison_ships_cache[:, 0] = self.planet_ships
        self.garrison_pre_combat_owner_cache[:, 0] = self.planet_owner
        self.garrison_pre_combat_ships_cache[:, 0] = self.planet_ships
        self.garrison_dirty_from = torch.zeros(self.P, dtype=torch.long, device=self.device)
    def _refresh_garrison_projection(self, slots: Tensor, *, requested_horizon: int | None = None) -> None:
        self._ensure_garrison_cache()
        assert self.fleet_buckets is not None
        assert self.garrison_owner_cache is not None
        assert self.garrison_ships_cache is not None
        assert self.garrison_dirty_from is not None
        p_idx = torch.unique(slots.reshape(-1).clamp(min=0, max=max(self.P - 1, 0)))
        if p_idx.numel() == 0:
            return
        dirty = self.garrison_dirty_from[p_idx]
        horizon = int(
            self.movement_horizon
            if requested_horizon is None
            else max(0, min(int(requested_horizon), int(self.movement_horizon)))
        )
        needs_refresh = dirty <= horizon
        if not bool(needs_refresh.any()):
            return
        p_idx = p_idx[needs_refresh]
        owner = self.planet_owner[p_idx].clone()
        ships = self.planet_ships[p_idx].clone()
        self.garrison_owner_cache[p_idx, 0] = owner
        self.garrison_ships_cache[p_idx, 0] = ships
        assert self.garrison_pre_combat_owner_cache is not None
        assert self.garrison_pre_combat_ships_cache is not None
        self.garrison_pre_combat_owner_cache[p_idx, 0] = owner
        self.garrison_pre_combat_ships_cache[p_idx, 0] = ships
        prod = self.planet_prod[p_idx]
        if horizon == 0:
            self.garrison_dirty_from[p_idx] = horizon + 1
            return
        self._fill_garrison_trajectory(
            p_idx=p_idx,
            init_owner=owner,
            init_ships=ships,
            prod=prod,
            horizon=horizon,
        )
        self.garrison_dirty_from[p_idx] = horizon + 1
    def _fill_garrison_trajectory(
        self,
        *,
        p_idx: Tensor,
        init_owner: Tensor,
        init_ships: Tensor,
        prod: Tensor,
        horizon: int,
    ) -> None:
        """Fill ``garrison_{owner,ships}_cache`` for steps ``1..horizon``.
        Decomposes the per-pair recurrence into two halves so the GPU does very
        little sequential work:
        - **Half A** (vectorized): compute the per-step combat survivor
          ``(top_owner, top1 - top2)`` over the player axis for all ``H`` steps in a
          single fused tensor op. The survivor is a pure function of that step's
          arrival vector and does not depend on the planet state, so this carries no
          inter-step dependency. Replaces ``H`` per-step ``topk`` calls with one.
        - **Half B** (sequential, branchless): walk ``k = 1..H`` advancing
          ``(state_owner, state_ships)``. Every operation is a fused ``where`` —
          there is no host sync (no ``bool(has_arrivals.any())``), no boolean
          indexing, and no per-step ``topk``. Each iteration is ~5 element-wise
          kernels over ``[N_complex]``, vs ~12 kernels + a host sync previously.
        Plus a closed-form fast path for "simple" pairs (no arrivals over the
        horizon and the planet stays alive throughout). For those pairs, owner is
        constant and ships grow linearly: ``ships[k] = ships[0] + prod * k``. We
        write the entire trajectory in one tensor assignment instead of iterating.
        Most planets in a typical match satisfy this, so the recurrent path runs
        on a small fraction of pairs.
        """
        assert self.fleet_buckets is not None
        assert self.garrison_owner_cache is not None
        assert self.garrison_ships_cache is not None
        assert self.garrison_pre_combat_owner_cache is not None
        assert self.garrison_pre_combat_ships_cache is not None
        H = int(horizon)
        N = int(p_idx.numel())
        if N == 0 or H == 0:
            return
        # ``alive_by_step[k, p]`` is the alive mask AT END of step ``k`` (= the
        # position frame for the k-th lookahead). For step k's transition we need
        # alive at the start (``alive_step[k-1]``) and at the end (``alive_step[k]``).
        alive_step = self.alive_by_step[:, p_idx].transpose(0, 1)  # [N, H+1]
        alive_before = alive_step[:, :H]                          # [N, H]
        alive_now = alive_step[:, 1:]                             # [N, H]
        # ``fleet_buckets[p, k, a]`` = ships from owner ``a`` arriving at step ``k+1``.
        arrivals = self.fleet_buckets[p_idx, :H, :]               # [N, H, A]
        # A pair is "simple" if no fleets ever arrive at this planet over the
        # horizon AND the planet stays alive throughout. For such pairs the
        # trajectory is purely additive: owner constant, ships grow by ``prod``
        # per step (or stay zero for neutral planets). Most planets in a typical
        # match fit this profile, so this is the big algorithmic win — these
        # pairs skip the per-step recurrence entirely.
        has_any_arrival = (arrivals > 0.0).any(dim=-1).any(dim=-1)  # [N]
        alive_all_true = alive_step.all(dim=1)                       # [N]
        simple_mask = (~has_any_arrival) & alive_all_true            # [N]
        # Cache the per-pair alive trajectory before we filter to complex pairs;
        # we'll need it for the tail-continuation step below.
        alive_step_full = alive_step
        # One host sync per refresh to count simple vs complex pairs.
        n_simple = int(simple_mask.sum().item())
        n_complex = N - n_simple
        if n_simple > 0:
            simple_p = p_idx[simple_mask]
            simple_owner = init_owner[simple_mask]
            simple_ships = init_ships[simple_mask]
            simple_prod = prod[simple_mask]
            # Production accrues only for owned planets; the ``(owner >= 0)`` factor
            # collapses neutral and dead planets to zero growth.
            owner_alive_factor = (simple_owner >= 0).to(dtype=simple_ships.dtype)
            k_range = torch.arange(1, H + 1, device=self.device, dtype=simple_ships.dtype)
            ships_traj = (
                simple_ships.unsqueeze(1)
                + simple_prod.unsqueeze(1)
                * owner_alive_factor.unsqueeze(1)
                * k_range.unsqueeze(0)
            )                                                         # [N_simple, H]
            owner_traj = simple_owner.unsqueeze(1).expand(-1, H)
            # One fused write per cache, covers every step 1..H simultaneously.
            self.garrison_owner_cache[simple_p, 1 : H + 1] = owner_traj
            self.garrison_ships_cache[simple_p, 1 : H + 1] = ships_traj
            # Simple-path pairs have no arrivals across the horizon, so
            # pre-combat state at every step equals the post-combat state.
            self.garrison_pre_combat_owner_cache[simple_p, 1 : H + 1] = owner_traj
            self.garrison_pre_combat_ships_cache[simple_p, 1 : H + 1] = ships_traj
        if n_complex == 0:
            return
        complex_mask = ~simple_mask
        cp = p_idx[complex_mask]
        arrivals_c = arrivals[complex_mask]                           # [N_c, H, A]
        alive_before_c = alive_before[complex_mask]                   # [N_c, H]
        alive_now_c = alive_now[complex_mask]                         # [N_c, H]
        alive_step_c = alive_step_full[complex_mask]                  # [N_c, H+1]
        state_owner = init_owner[complex_mask].clone()                # [N_c]
        state_ships = init_ships[complex_mask].clone()                # [N_c]
        prod_c = prod[complex_mask]                                   # [N_c]
        # Half A: per-step (top1 - top2) survivor over the player axis. No
        # cross-step dependency, so it runs in one fused op rather than ``H``
        # times in the inner loop.
        A = int(arrivals_c.shape[-1])
        if A >= 2:
            top2 = arrivals_c.topk(k=2, dim=-1)
            top_ships_traj = top2.values[..., 0]
            second_ships_traj = top2.values[..., 1]
            top_owner_traj = top2.indices[..., 0].to(dtype=torch.long)
        else:
            top_ships_traj, top_owner_traj = arrivals_c.max(dim=-1)
            second_ships_traj = torch.zeros_like(top_ships_traj)
            top_owner_traj = top_owner_traj.to(dtype=torch.long)
        # Ties leave no survivor (mutual annihilation). Where both top values
        # are zero (no arrivals at this step), ``survivor_ships`` is also zero
        # and ``has_combat`` will mask the step out below.
        tied = top_ships_traj == second_ships_traj
        survivor_ships_traj = torch.where(
            tied,
            torch.zeros_like(top_ships_traj),
            (top_ships_traj - second_ships_traj).clamp(min=0.0),
        )                                                          # [N_c, H]
        survivor_owner_traj = top_owner_traj                       # [N_c, H]
        # Scalar broadcast templates for the ``where``-based death reset; using
        # scalars keeps each per-step ``where`` to a single small kernel.
        zero_ships_scalar = torch.zeros((), dtype=state_ships.dtype, device=self.device)
        neg_one_owner_scalar = torch.full((), -1, dtype=state_owner.dtype, device=self.device)
        zero_prod_scalar = torch.zeros((), dtype=prod_c.dtype, device=self.device)
        # Horizon-trim optimization: identify the latest step at which ANY complex
        # pair has a structural transition. Beyond that step every pair's
        # trajectory is determined purely by production accumulation, so we can
        # replace the rest of the H-step recurrence with one closed-form tensor
        # write (analogous to the simple-pair fast path). Two kinds of structural
        # transitions can change a pair's state:
        #   - a non-tied combat survivor lands while the planet is alive
        #     (``has_combat = (s_ships > 0) & alive_now``);
        #   - the planet's alive state flips (death or respawn) at this step.
        combat_event_per_step = (survivor_ships_traj > 0.0) & alive_now_c   # [N_c, H]
        alive_change_per_step = alive_before_c != alive_now_c                # [N_c, H]
        any_event_per_step = (combat_event_per_step | alive_change_per_step).any(dim=0)  # [H]
        # Map each step k ∈ [1, H] to itself if there's an event there, else 0.
        # The max collapses to the largest ``k`` with any event, or 0 if none.
        arange_h = torch.arange(1, H + 1, device=self.device, dtype=torch.long)
        k_last_tensor = torch.where(
            any_event_per_step,
            arange_h,
            torch.zeros_like(arange_h),
        ).max()
        # One host sync per refresh: we need ``k_last`` on the host to size the
        # Python loop. The win from shrinking the loop dwarfs the sync cost.
        k_last = int(k_last_tensor.item())
        loop_iters = max(0, k_last)
        tail_steps = H - loop_iters
        if loop_iters > 0:
            # Half B: branchless H-step recurrence. The ``(state_owner, state_ships)``
            # pair has a real cross-step dependency — an attacker capturing the planet
            # at step k flips who produces in subsequent steps — so we must walk
            # ``loop_iters`` sequentially. Each iteration is fully branchless: no host
            # sync, no boolean indexing, no ``topk``. Just element-wise ``where``s
            # over ``[N_c]``.
            for k in range(1, loop_iters + 1):
                a_before = alive_before_c[:, k - 1]
                a_now = alive_now_c[:, k - 1]
                s_owner = survivor_owner_traj[:, k - 1]
                s_ships = survivor_ships_traj[:, k - 1]
                # Production: owned planets that were alive at the start of this step.
                produces = a_before & (state_owner >= 0)
                state_ships = state_ships + torch.where(produces, prod_c, zero_prod_scalar)
                # Snapshot pre-combat state: this is what an attacker arriving
                # at step ``k`` will face from the planet itself, before any
                # same-step attacker combat is applied. Captured here so a
                # planner can synthesize "what if I also arrive this turn?"
                # using the engine's combat rule.
                pre_owner = torch.where(a_now, state_owner, neg_one_owner_scalar)
                pre_ships = torch.where(a_now, state_ships, zero_ships_scalar)
                self.garrison_pre_combat_owner_cache[cp, k] = pre_owner
                self.garrison_pre_combat_ships_cache[cp, k] = pre_ships
                # Combat against the precomputed step-k survivor. Three cases collapse
                # into two ``where`` chains masked by ``has_combat``:
                #   same owner: state_ships += s_ships  (reinforcement)
                #   ~same & state_ships <  s_ships: planet flips, ships = s_ships - state_ships
                #   ~same & state_ships >= s_ships: garrison reduced by s_ships
                has_combat = (s_ships > 0.0) & a_now
                same = state_owner == s_owner
                diff = state_ships - s_ships  # signed; |diff| is the post-combat ships count
                attacker_wins = (~same) & (diff < 0.0)
                combat_ships = torch.where(same, state_ships + s_ships, diff.abs())
                combat_owner = torch.where(attacker_wins, s_owner, state_owner)
                state_ships = torch.where(has_combat, combat_ships, state_ships)
                state_owner = torch.where(has_combat, combat_owner, state_owner)
                # End-of-step death reset: if the planet despawns this step it has
                # no owner and no garrison from now on.
                state_owner = torch.where(a_now, state_owner, neg_one_owner_scalar)
                state_ships = torch.where(a_now, state_ships, zero_ships_scalar)
                self.garrison_owner_cache[cp, k] = state_owner
                self.garrison_ships_cache[cp, k] = state_ships
        if tail_steps > 0:
            # By construction of ``k_last``, no complex pair has a structural event
            # at any step in ``(k_last, H]``: alive is constant, no combat survivors,
            # no captures. So the trajectory across the tail is closed-form:
            #   ships[k] = state_ships + prod * (k - k_last) * (alive AND owned)
            #   owner[k] = state_owner    (constant)
            # We still need to apply the "pending" death reset for pairs whose
            # ``alive_step[k_last]`` is False. When ``k_last >= 1`` the loop's last
            # iteration already did this; when ``k_last == 0`` we apply it here so
            # the closed-form formula matches the original loop's output.
            alive_at_k_last = alive_step_c[:, k_last]                  # [N_c]
            state_owner = torch.where(alive_at_k_last, state_owner, neg_one_owner_scalar)
            state_ships = torch.where(alive_at_k_last, state_ships, zero_ships_scalar)
            # Production multiplier: 1 only for pairs that are alive AND owned at
            # ``k_last`` (and therefore for the entire tail by definition).
            owner_alive_factor = (
                (state_owner >= 0).to(dtype=state_ships.dtype)
                * alive_at_k_last.to(dtype=state_ships.dtype)
            )                                                          # [N_c]
            # ``dk_range[i]`` = i + 1, the offset from ``k_last`` to step ``k_last+1+i``.
            dk_range = torch.arange(
                1, tail_steps + 1, device=self.device, dtype=state_ships.dtype
            )                                                          # [tail_steps]
            ships_traj_tail = (
                state_ships.unsqueeze(1)
                + prod_c.unsqueeze(1)
                * owner_alive_factor.unsqueeze(1)
                * dk_range.unsqueeze(0)
            )                                                          # [N_c, tail_steps]
            owner_traj_tail = state_owner.unsqueeze(1).expand(-1, tail_steps)
            self.garrison_owner_cache[cp, k_last + 1 : H + 1] = owner_traj_tail
            self.garrison_ships_cache[cp, k_last + 1 : H + 1] = ships_traj_tail
            # Tail has no structural events (no combat, no death), so the
            # pre-combat state at every tail step equals the post-combat
            # state — production only.
            self.garrison_pre_combat_owner_cache[cp, k_last + 1 : H + 1] = owner_traj_tail
            self.garrison_pre_combat_ships_cache[cp, k_last + 1 : H + 1] = ships_traj_tail
    def _roll_garrison_projection(self) -> None:
        if (
            self.garrison_owner_cache is None
            or self.garrison_ships_cache is None
            or self.garrison_pre_combat_owner_cache is None
            or self.garrison_pre_combat_ships_cache is None
            or self.garrison_dirty_from is None
        ):
            return
        horizon = int(self.movement_horizon)
        if horizon > 0:
            self.garrison_owner_cache[:, :-1] = self.garrison_owner_cache[:, 1:].clone()
            self.garrison_ships_cache[:, :-1] = self.garrison_ships_cache[:, 1:].clone()
            self.garrison_pre_combat_owner_cache[:, :-1] = (
                self.garrison_pre_combat_owner_cache[:, 1:].clone()
            )
            self.garrison_pre_combat_ships_cache[:, :-1] = (
                self.garrison_pre_combat_ships_cache[:, 1:].clone()
            )
            self.garrison_dirty_from = (self.garrison_dirty_from - 1).clamp(min=0)
            self.garrison_dirty_from = torch.minimum(
                self.garrison_dirty_from,
                torch.full_like(self.garrison_dirty_from, horizon),
            )
        else:
            self.garrison_dirty_from[:] = 0
    def _refresh_garrison_base(self, built: dict[str, Tensor]) -> None:
        owner = built["owner"].to(device=self.device, dtype=torch.long)
        ships = built["ships"].to(device=self.device, dtype=self.dtype)
        prod = built["prod"].to(device=self.device, dtype=self.dtype)
        prod_changed = tuple(self.planet_prod.shape) != tuple(prod.shape) or (self.planet_prod != prod)
        self.planet_owner = owner
        self.planet_ships = ships
        self.planet_prod = prod
        if self.garrison_owner_cache is None or self.garrison_ships_cache is None or self.garrison_dirty_from is None:
            return
        base_changed = (
            (self.garrison_owner_cache[:, 0] != owner)
            | (self.garrison_ships_cache[:, 0] != ships)
        )
        self.garrison_owner_cache[:, 0] = owner
        self.garrison_ships_cache[:, 0] = ships
        if self.garrison_pre_combat_owner_cache is not None:
            self.garrison_pre_combat_owner_cache[:, 0] = owner
        if self.garrison_pre_combat_ships_cache is not None:
            self.garrison_pre_combat_ships_cache[:, 0] = ships
        if bool(base_changed.any()):
            self.garrison_dirty_from[base_changed] = 0
        if isinstance(prod_changed, Tensor) and bool(prod_changed.any()):
            self.garrison_dirty_from[prod_changed] = torch.minimum(
                self.garrison_dirty_from[prod_changed],
                torch.ones_like(self.garrison_dirty_from[prod_changed]),
            )
        elif not isinstance(prod_changed, Tensor) and prod_changed:
            self.garrison_dirty_from[:] = torch.minimum(
                self.garrison_dirty_from,
                torch.ones_like(self.garrison_dirty_from),
            )
    def _mark_garrison_dirty(self, planet_idx: Tensor, start_step: Tensor | int) -> None:
        if self.garrison_dirty_from is None:
            return
        p = planet_idx.to(device=self.device, dtype=torch.long)
        if isinstance(start_step, int):
            start = torch.full((), int(start_step), dtype=torch.long, device=self.device)
        else:
            start = start_step.to(device=self.device, dtype=torch.long)
        p, start = torch.broadcast_tensors(p, start)
        p = p.reshape(-1)
        start = start.reshape(-1)
        if p.numel() == 0:
            return
        start = start.clamp(min=0, max=int(self.movement_horizon))
        valid = (p >= 0) & (p < self.P)
        if not bool(valid.any()):
            return
        p = p[valid]
        start = start[valid]
        flat = self.garrison_dirty_from
        unique_idx, inverse = torch.unique(p, return_inverse=True)
        if unique_idx.numel() == p.numel():
            flat[unique_idx] = torch.minimum(flat[unique_idx], start)
            return
        sentinel = int(self.movement_horizon) + 1
        candidate = torch.full((unique_idx.shape[0],), sentinel, dtype=torch.long, device=self.device)
        candidate.scatter_reduce_(0, inverse, start, reduce="amin", include_self=True)
        flat[unique_idx] = torch.minimum(flat[unique_idx], candidate)
    def _mark_garrison_dirty_all(self, start_step: int) -> None:
        if self.garrison_dirty_from is None:
            return
        self.garrison_dirty_from = torch.minimum(
            self.garrison_dirty_from,
            torch.full_like(self.garrison_dirty_from, int(start_step)),
        )
    def _init_fleet_tracking(self, obs_tensors: dict, *, reset_ledger: bool) -> None:
        _ = reset_ledger
        player_count = _resolve_player_count(obs_tensors, self.player_count)
        self.player_count = int(player_count)
        self.fleet_buckets = torch.zeros(
            self.P,
            int(self.movement_horizon),
            int(player_count),
            dtype=self.dtype,
            device=self.device,
        )
        step = obs_tensors["step"].to(device=self.device, dtype=torch.long)
        self.fleet_last_step = step.detach().clone()
        M = max(1, int(self.max_tracked_fleets))
        self.max_tracked_fleets = M
        self.tracked_fleet_ids = torch.full((M,), -1, dtype=torch.long, device=self.device)
        self.tracked_fleet_eta = torch.zeros((M,), dtype=torch.long, device=self.device)
        self.tracked_fleet_target_slot = torch.full((M,), -1, dtype=torch.long, device=self.device)
        self.tracked_fleet_owner = torch.zeros((M,), dtype=torch.long, device=self.device)
        self.tracked_fleet_ships = torch.zeros((M,), dtype=self.dtype, device=self.device)
        if self.garrison_dirty_from is not None:
            self.garrison_dirty_from[:] = torch.minimum(
                self.garrison_dirty_from,
                torch.full_like(self.garrison_dirty_from, 1),
            )
    def _clear_tracked_rows(self) -> None:
        if (
            self.tracked_fleet_ids is None
            or self.tracked_fleet_eta is None
            or self.tracked_fleet_target_slot is None
            or self.tracked_fleet_owner is None
            or self.tracked_fleet_ships is None
        ):
            return
        self.tracked_fleet_ids[:] = -1
        self.tracked_fleet_eta[:] = 0
        self.tracked_fleet_target_slot[:] = -1
        self.tracked_fleet_owner[:] = 0
        self.tracked_fleet_ships[:] = 0.0
    def _ledger_bulk_insert(
        self,
        fleet_ids: Tensor,
        eta_remaining: Tensor,
        target_slots: Tensor,
        owners: Tensor,
        ships: Tensor,
    ) -> None:
        if fleet_ids.numel() == 0:
            return
        assert self.tracked_fleet_ids is not None
        assert self.tracked_fleet_eta is not None
        assert self.tracked_fleet_target_slot is not None
        assert self.tracked_fleet_owner is not None
        assert self.tracked_fleet_ships is not None
        M = int(self.tracked_fleet_ids.shape[0])
        fleet_ids = fleet_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        eta_remaining = eta_remaining.to(device=self.device, dtype=torch.long).reshape(-1)
        target_slots = target_slots.to(device=self.device, dtype=torch.long).reshape(-1)
        owners = owners.to(device=self.device, dtype=torch.long).reshape(-1)
        ships = ships.to(device=self.device, dtype=self.dtype).reshape(-1)
        valid_rows = fleet_ids >= 0
        if not bool(valid_rows.any()):
            return
        fleet_ids = fleet_ids[valid_rows]
        eta_remaining = eta_remaining[valid_rows]
        target_slots = target_slots[valid_rows]
        owners = owners[valid_rows]
        ships = ships[valid_rows]
        n = int(fleet_ids.numel())
        empty_mask = self.tracked_fleet_ids == -1
        empty_count = int(empty_mask.sum().item())
        if n > empty_count:
            occupied_count = M - empty_count
            self._grow_ledger_capacity(occupied_count + n)
            assert self.tracked_fleet_ids is not None
            empty_mask = self.tracked_fleet_ids == -1
        # Place the rows into the first ``n`` empty ledger slots, ascending —
        # which preserves input order (each row keeps its emission rank).
        empty_slots = torch.nonzero(empty_mask, as_tuple=True)[0]
        slot_idx = empty_slots[:n]
        self.tracked_fleet_ids[slot_idx] = fleet_ids
        self.tracked_fleet_eta[slot_idx] = eta_remaining
        self.tracked_fleet_target_slot[slot_idx] = target_slots
        self.tracked_fleet_owner[slot_idx] = owners
        self.tracked_fleet_ships[slot_idx] = ships
    def _grow_ledger_capacity(self, required_capacity: int) -> None:
        if (
            self.tracked_fleet_ids is None
            or self.tracked_fleet_eta is None
            or self.tracked_fleet_target_slot is None
            or self.tracked_fleet_owner is None
            or self.tracked_fleet_ships is None
        ):
            return
        old_capacity = int(self.tracked_fleet_ids.shape[0])
        target_capacity = max(int(required_capacity), old_capacity)
        if target_capacity <= old_capacity:
            return
        new_capacity = max(target_capacity, old_capacity * 2)
        old_ids = self.tracked_fleet_ids
        old_eta = self.tracked_fleet_eta
        old_tgt = self.tracked_fleet_target_slot
        old_owner = self.tracked_fleet_owner
        old_ships = self.tracked_fleet_ships
        self.tracked_fleet_ids = torch.full((new_capacity,), -1, dtype=torch.long, device=self.device)
        self.tracked_fleet_eta = torch.zeros((new_capacity,), dtype=torch.long, device=self.device)
        self.tracked_fleet_target_slot = torch.full((new_capacity,), -1, dtype=torch.long, device=self.device)
        self.tracked_fleet_owner = torch.zeros((new_capacity,), dtype=torch.long, device=self.device)
        self.tracked_fleet_ships = torch.zeros((new_capacity,), dtype=self.dtype, device=self.device)
        self.tracked_fleet_ids[:old_capacity] = old_ids
        self.tracked_fleet_eta[:old_capacity] = old_eta
        self.tracked_fleet_target_slot[:old_capacity] = old_tgt
        self.tracked_fleet_owner[:old_capacity] = old_owner
        self.tracked_fleet_ships[:old_capacity] = old_ships
    def _ledger_decrement_and_expire(self) -> None:
        if (
            self.tracked_fleet_ids is None
            or self.tracked_fleet_eta is None
            or self.tracked_fleet_target_slot is None
            or self.tracked_fleet_owner is None
            or self.tracked_fleet_ships is None
        ):
            return
        valid = self.tracked_fleet_ids >= 0
        eta = torch.where(valid, self.tracked_fleet_eta - 1, self.tracked_fleet_eta)
        expire = valid & (eta <= 0)
        self.tracked_fleet_eta = eta
        self.tracked_fleet_ids = torch.where(expire, torch.full_like(self.tracked_fleet_ids, -1), self.tracked_fleet_ids)
        self.tracked_fleet_eta = torch.where(expire, torch.zeros_like(self.tracked_fleet_eta), self.tracked_fleet_eta)
        self.tracked_fleet_target_slot = torch.where(
            expire,
            torch.full_like(self.tracked_fleet_target_slot, -1),
            self.tracked_fleet_target_slot,
        )
        self.tracked_fleet_owner = torch.where(
            expire,
            torch.zeros_like(self.tracked_fleet_owner),
            self.tracked_fleet_owner,
        )
        self.tracked_fleet_ships = torch.where(
            expire,
            torch.zeros_like(self.tracked_fleet_ships),
            self.tracked_fleet_ships,
        )
    def _roll_fleet_buckets_phase1(self, current_step: Tensor) -> None:
        if self.fleet_buckets is None or self.fleet_last_step is None:
            return
        step = current_step.to(device=self.device, dtype=torch.long)
        delta = step - self.fleet_last_step.to(device=self.device, dtype=torch.long)
        horizon = int(self.movement_horizon)
        reset = bool((delta < 0) | (step <= 0))
        if reset:
            self.fleet_buckets[:] = 0.0
            self._clear_tracked_rows()
            self._mark_garrison_dirty_all(1)
        rolled_once = (not reset) and bool(delta == 1)
        if rolled_once and horizon > 0:
            self.fleet_buckets[:, :-1, :] = self.fleet_buckets[:, 1:, :].clone()
            self.fleet_buckets[:, -1, :] = 0.0
            self._ledger_decrement_and_expire()
            self._mark_garrison_dirty_all(1)
        delta_bad = (not reset) and bool(delta > 1)
        if delta_bad:
            self._reset_fleet_tracking()
        self.fleet_last_step = step.detach().clone()
    def _reset_fleet_tracking(self) -> None:
        if self.fleet_buckets is None:
            return
        self.fleet_buckets[:] = 0.0
        self._clear_tracked_rows()
        self._mark_garrison_dirty_all(1)
    def _ingest_obs_fleets(self, obs_tensors: dict) -> None:
        if self.fleet_buckets is None or self.tracked_fleet_ids is None or int(self.movement_horizon) <= 0:
            return
        fleets = obs_tensors["fleets"].to(device=self.device, dtype=self.dtype)
        fleet_ids = fleets[..., 0].to(dtype=torch.long)
        alive = fleet_ids >= 0
        # Pairwise compare every observed fleet id against every ledger row id;
        # shape ``[F_obs, M_ledger]`` collapsed by ``any(dim=-1)``. New (untracked)
        # alive fleets get their arrival estimated and recorded.
        tracked = (fleet_ids.unsqueeze(1) == self.tracked_fleet_ids.unsqueeze(0)).any(dim=1)
        process_mask = alive & ~tracked
        n_alive = int(alive.sum().item())
        n_tracked = int((alive & tracked).sum().item())
        n_to_process = n_alive - n_tracked
        if n_to_process == 0:
            return
        fleet_slot = torch.where(process_mask)[0]
        proc_ids = fleet_ids[fleet_slot]
        estimate = _estimate_new_fleet_arrivals(movement=self, obs_fleets=fleets, fleet_slot=fleet_slot)
        valid_owner = (estimate["owner"] >= 0) & (estimate["owner"] < int(self.fleet_buckets.shape[2]))
        valid_hit = estimate["has_hit"] & valid_owner
        if not bool(valid_hit.any()):
            return
        buckets = self._require_fleet_buckets()
        buckets.index_put_(
            (
                estimate["target_slot"][valid_hit],
                estimate["eta_index"][valid_hit],
                estimate["owner"][valid_hit],
            ),
            estimate["ships"][valid_hit],
            accumulate=True,
        )
        self._mark_garrison_dirty(
            estimate["target_slot"][valid_hit],
            estimate["eta_index"][valid_hit] + 1,
        )
        eta_remaining = estimate["eta_index"][valid_hit].to(dtype=torch.long) + 1
        self._ledger_bulk_insert(
            proc_ids[valid_hit],
            eta_remaining,
            estimate["target_slot"][valid_hit],
            estimate["owner"][valid_hit],
            estimate["ships"][valid_hit],
        )
    def _reconcile_obs_fleets(self, obs_tensors: dict) -> None:
        """Drop ledger entries whose fleet is no longer in obs.
        ``record_fleet_arrivals`` writes a fleet's predicted arrival into both
        ``fleet_buckets`` and the tracked-fleet ledger at launch time. If the
        engine destroys the fleet before it arrives (sun crossing, OOB,
        unintended planet collision), the fleet disappears from ``obs.fleets``
        but neither ``_ingest_obs_fleets`` nor ``_ledger_decrement_and_expire``
        knows to evict it — ingest only adds, decrement only fires at eta=0.
        This pass walks ``tracked_fleet_ids``, checks each non-empty entry
        against the current ``obs.fleets[..., 0]``, and for any phantom
        (in-ledger, in-flight, not-in-obs) subtracts its recorded ships from
        ``fleet_buckets`` at the entry's stored ``(target_slot, eta-1, owner)``
        and clears the row. Marks the touched garrison cells dirty so the next
        ``garrison_status`` query rebuilds them.
        """
        if (
            self.fleet_buckets is None
            or self.tracked_fleet_ids is None
            or self.tracked_fleet_eta is None
            or self.tracked_fleet_target_slot is None
            or self.tracked_fleet_owner is None
            or self.tracked_fleet_ships is None
            or int(self.movement_horizon) <= 0
        ):
            return
        obs_ids = obs_tensors["fleets"][..., 0].to(device=self.device, dtype=torch.long)  # [F]
        in_flight = (self.tracked_fleet_ids >= 0) & (self.tracked_fleet_eta > 0)
        if not bool(in_flight.any()):
            return
        # ``[M, F]`` pairwise compare; ``any(dim=-1)`` gives ledger-side in-obs.
        match = (self.tracked_fleet_ids.unsqueeze(1) == obs_ids.unsqueeze(0)).any(dim=1)
        phantom = in_flight & ~match
        if not bool(phantom.any()):
            return
        m_idx = torch.where(phantom)[0]
        h_idx = (self.tracked_fleet_eta[m_idx] - 1).clamp(min=0)
        P = int(self.fleet_buckets.shape[0])
        H = int(self.fleet_buckets.shape[1])
        A = int(self.fleet_buckets.shape[2])
        in_horizon = h_idx < H
        if not bool(in_horizon.any()):
            self.tracked_fleet_ids[m_idx] = -1
            self.tracked_fleet_eta[m_idx] = 0
            self.tracked_fleet_target_slot[m_idx] = -1
            self.tracked_fleet_owner[m_idx] = 0
            self.tracked_fleet_ships[m_idx] = 0.0
            return
        m_sel = m_idx[in_horizon]
        h_sel = h_idx[in_horizon]
        slots = self.tracked_fleet_target_slot[m_sel].clamp(min=0, max=max(P - 1, 0))
        owners = self.tracked_fleet_owner[m_sel].clamp(min=0, max=max(A - 1, 0))
        ships = self.tracked_fleet_ships[m_sel]
        self.fleet_buckets.index_put_(
            (slots, h_sel, owners),
            -ships,
            accumulate=True,
        )
        # ``h_sel`` is the bucket index; ``k = h_sel + 1`` is the corresponding
        # arrival step in garrison-projection coordinates.
        self._mark_garrison_dirty(slots, h_sel + 1)
        # Clear every phantom row (in-horizon and out-of-horizon alike).
        self.tracked_fleet_ids[m_idx] = -1
        self.tracked_fleet_eta[m_idx] = 0
        self.tracked_fleet_target_slot[m_idx] = -1
        self.tracked_fleet_owner[m_idx] = 0
        self.tracked_fleet_ships[m_idx] = 0.0
    def _require_fleet_buckets(self) -> Tensor:
        if self.fleet_buckets is None:
            raise RuntimeError("PlanetMovement fleet tracking is not enabled")
        return self.fleet_buckets
    def _k_index(self, k: int) -> int:
        if k < 0 or k > int(self.movement_horizon):
            raise IndexError(f"k must be in [0, {self.movement_horizon}], got {k}")
        return int(k)
    def _copy_from(self, other: "PlanetMovement") -> None:
        self.x = other.x
        self.y = other.y
        self.alive_by_step = other.alive_by_step
        self.planet_ids = other.planet_ids
        self.radii = other.radii
        self.planet_owner = other.planet_owner
        self.planet_ships = other.planet_ships
        self.planet_prod = other.planet_prod
        self.base_step = other.base_step
        self.comet_planet_ids = other.comet_planet_ids
        self.comet_path_index = other.comet_path_index
        self.movement_horizon = other.movement_horizon
        self.drift_epsilon = other.drift_epsilon
        self.track_fleets = other.track_fleets
        self.player_count = other.player_count
        self.max_tracked_fleets = other.max_tracked_fleets
        self.fleet_buckets = other.fleet_buckets
        self.fleet_last_step = other.fleet_last_step
        self.tracked_fleet_ids = other.tracked_fleet_ids
        self.tracked_fleet_eta = other.tracked_fleet_eta
        self.tracked_fleet_target_slot = other.tracked_fleet_target_slot
        self.tracked_fleet_owner = other.tracked_fleet_owner
        self.tracked_fleet_ships = other.tracked_fleet_ships
        self.garrison_owner_cache = other.garrison_owner_cache
        self.garrison_ships_cache = other.garrison_ships_cache
        self.garrison_dirty_from = other.garrison_dirty_from
def _resolve_player_count(obs_tensors: dict, player_count: int | None) -> int:
    if player_count is not None:
        if int(player_count) not in (2, 4):
            raise ValueError("player_count must be 2 or 4")
        return int(player_count)
    metadata_count = obs_tensors.get("player_count")
    if metadata_count is not None:
        count = int(metadata_count.flatten()[0].item()) if isinstance(metadata_count, Tensor) else int(metadata_count)
        if count not in (2, 4):
            raise ValueError("player_count metadata must be 2 or 4")
        return count
    planets = obs_tensors["planets"]
    fleets = obs_tensors["fleets"]
    planet_alive = planets[..., 0] >= 0
    fleet_alive = fleets[..., 0] >= 0
    owner_values = []
    if bool(planet_alive.any()):
        owner_values.append(planets[..., 1][planet_alive].to(dtype=torch.long))
    if bool(fleet_alive.any()):
        owner_values.append(fleets[..., 1][fleet_alive].to(dtype=torch.long))
    if not owner_values:
        return 2
    owners = torch.cat(owner_values)
    owners = owners[owners >= 0]
    if owners.numel() == 0:
        return 2
    return 4 if int(owners.max().item()) >= 2 else 2
def _estimate_new_fleet_arrivals(
    *,
    movement: PlanetMovement,
    obs_fleets: Tensor,
    fleet_slot: Tensor,
) -> dict[str, Tensor]:
    N = int(fleet_slot.numel())
    device = movement.device
    dtype = movement.dtype
    H = int(movement.movement_horizon)
    P = int(movement.P)
    if N == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_bool = torch.empty(0, dtype=torch.bool, device=device)
        empty_float = torch.empty(0, dtype=dtype, device=device)
        return {
            "owner": empty_long,
            "target_slot": empty_long,
            "eta_index": empty_long,
            "has_hit": empty_bool,
            "ships": empty_float,
        }
    rows = obs_fleets[fleet_slot]
    owner = rows[:, 1].to(dtype=torch.long)
    x = rows[:, 2].to(dtype=dtype)
    y = rows[:, 3].to(dtype=dtype)
    angle = rows[:, 4].to(dtype=dtype)
    ships = rows[:, 6].to(dtype=dtype)
    times = torch.arange(1, H + 1, dtype=dtype, device=device).view(1, H)
    speed = fleet_speed(ships).clamp(min=1e-6)
    ux = torch.cos(angle)
    uy = torch.sin(angle)
    old_x = x.view(N, 1) + ux.view(N, 1) * speed.view(N, 1) * (times - 1.0)
    old_y = y.view(N, 1) + uy.view(N, 1) * speed.view(N, 1) * (times - 1.0)
    new_x = x.view(N, 1) + ux.view(N, 1) * speed.view(N, 1) * times
    new_y = y.view(N, 1) + uy.view(N, 1) * speed.view(N, 1) * times
    in_bounds = (new_x >= 0.0) & (new_x <= BOARD_SIZE) & (new_y >= 0.0) & (new_y <= BOARD_SIZE)
    sun_dist_sq = _point_to_segment_distance_sq(
        torch.full_like(new_x, CENTER),
        torch.full_like(new_y, CENTER),
        old_x,
        old_y,
        new_x,
        new_y,
    )
    env_kill = (~in_bounds) | (sun_dist_sq < (SUN_RADIUS * SUN_RADIUS))
    planet_x = movement.x.unsqueeze(0).expand(N, H + 1, P)
    planet_y = movement.y.unsqueeze(0).expand(N, H + 1, P)
    planet_alive = movement.alive_by_step.unsqueeze(0).expand(N, H + 1, P)
    radii = movement.radii.unsqueeze(0).expand(N, P).to(dtype=dtype)
    old_px = planet_x[:, :-1, :]
    old_py = planet_y[:, :-1, :]
    new_px = planet_x[:, 1:, :]
    new_py = planet_y[:, 1:, :]
    alive_old = planet_alive[:, :-1, :]
    check_collision = alive_old & (old_px >= 0.0) & (old_py >= 0.0)
    swept_collides = _swept_pair_hit_mask_mv(
        old_x.unsqueeze(2),
        old_y.unsqueeze(2),
        new_x.unsqueeze(2),
        new_y.unsqueeze(2),
        old_px,
        old_py,
        new_px,
        new_py,
        radii.view(N, 1, P),
    ) & check_collision
    step_raw_has_hit = swept_collides.any(dim=2)
    hit_rank = swept_collides.to(torch.int32).cumsum(dim=2)
    first_hit = swept_collides & (hit_rank == 1)
    step_hit_slot = first_hit.to(torch.int64).argmax(dim=2)
    step_hit_slot = step_hit_slot.where(step_raw_has_hit, torch.full_like(step_hit_slot, -1))
    # Per-step ordering mirrors engine semantics: planet collision first,
    # out-of-bounds/sun checks only if no planet collision happened this step.
    # Vectorized active-mask propagation: a fleet is alive at the start of
    # turn t iff no kill event (planet hit OR env kill) has fired at any
    # turn τ < t. ``cummax`` along the time axis gives the inclusive OR;
    # shifting right by one (prepending alive=True) yields the exclusive form.
    kill_event = step_raw_has_hit | env_kill
    cum_kill_inclusive = kill_event.cummax(dim=1).values
    alive_before_t = torch.cat(
        [
            torch.ones((N, 1), dtype=torch.bool, device=device),
            ~cum_kill_inclusive[:, :-1],
        ],
        dim=1,
    )
    step_has_hit = step_raw_has_hit & alive_before_t
    has_hit = step_has_hit.any(dim=1)
    eta_index = step_has_hit.to(torch.int64).argmax(dim=1)
    target_slot = step_hit_slot.gather(1, eta_index.view(N, 1)).squeeze(1).clamp(min=0, max=max(P - 1, 0))
    return {
        "owner": owner,
        "target_slot": target_slot,
        "eta_index": eta_index,
        "has_hit": has_hit,
        "ships": ships,
    }
def _point_to_segment_distance_sq(px: Tensor, py: Tensor, x1: Tensor, y1: Tensor, x2: Tensor, y2: Tensor) -> Tensor:
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    t = ((px - x1) * dx + (py - y1) * dy) / safe_denom
    t = t.clamp(0.0, 1.0)
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return (px - proj_x) ** 2 + (py - proj_y) ** 2
def _swept_pair_hit_mask_mv(
    ax: Tensor,
    ay: Tensor,
    bx: Tensor,
    by: Tensor,
    p0x: Tensor,
    p0y: Tensor,
    p1x: Tensor,
    p1y: Tensor,
    r: Tensor,
) -> Tensor:
    """Broadcasted swept-pair overlap check for moving fleet/planet pairs."""
    d0x = ax - p0x
    d0y = ay - p0y
    dvx = (bx - ax) - (p1x - p0x)
    dvy = (by - ay) - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    near_static = a < 1e-12
    c_hit = c <= 0.0
    disc = b * b - 4.0 * a * c
    has_root = disc >= 0.0
    safe_a = torch.where(near_static, torch.ones_like(a), a)
    sq = torch.sqrt(torch.clamp(disc, min=0.0))
    t1 = (-b - sq) / (2.0 * safe_a)
    t2 = (-b + sq) / (2.0 * safe_a)
    quad_hit = has_root & (t2 >= 0.0) & (t1 <= 1.0)
    return torch.where(near_static, c_hit, quad_hit)
def _build_future_from_obs(
    obs_tensors: dict,
    movement_horizon: int,
    *,
    offsets: Tensor | None = None,
) -> dict[str, Tensor]:
    """Build planet/comet positions at the requested integer step offsets.
    By default builds the full trajectory ``offsets = arange(H+1)`` (output
    ``x/y/alive_by_step`` shape ``[H+1, P]``). Callers that only need a
    subset of frames (e.g. just the new last frame ``H`` on the roll-only
    update path) can pass ``offsets`` as a 1D long tensor; the output's
    first axis matches its length.
    """
    obs = parse_obs(obs_tensors)
    H = int(movement_horizon)
    planets = obs_tensors["planets"]
    dtype = planets.dtype
    device = planets.device
    P, _ = planets.shape
    planet_ids = planets[..., 0].long()
    radii = planets[..., 4].to(dtype=dtype)
    owner = planets[..., 1].to(device=device, dtype=torch.long)
    owner = torch.where(obs.alive, owner, torch.full_like(owner, -1))
    ships = planets[..., 5].to(device=device, dtype=dtype)
    prod = planets[..., 6].to(device=device, dtype=dtype)
    step = obs.step.to(device=device, dtype=torch.long)
    if offsets is None:
        offsets_long = torch.arange(H + 1, dtype=torch.long, device=device)
    else:
        offsets_long = offsets.to(device=device, dtype=torch.long).reshape(-1)
    M = int(offsets_long.shape[0])
    offsets_d = offsets_long.to(dtype=dtype)
    future_phase = orbit_phase_index_from_obs_step(
        obs.step.to(dtype=dtype) + offsets_d
    ).to(device=device, dtype=dtype)                                          # [M]
    angle = (
        obs.orb_a0.to(dtype=dtype).view(1, P)
        + obs.angvel.to(dtype=dtype) * future_phase.view(M, 1)
    )                                                                         # [M, P]
    orb_x = CENTER + obs.orb_r.to(dtype=dtype).view(1, P) * torch.cos(angle)
    orb_y = CENTER + obs.orb_r.to(dtype=dtype).view(1, P) * torch.sin(angle)
    is_orbiting = obs.is_orbiting.view(1, P)
    x = torch.where(
        is_orbiting,
        orb_x,
        obs.x.to(dtype=dtype).view(1, P).expand(M, P),
    ).contiguous()
    y = torch.where(
        is_orbiting,
        orb_y,
        obs.y.to(dtype=dtype).view(1, P).expand(M, P),
    ).contiguous()
    alive_by_step = obs.alive.view(1, P).expand(M, P).clone()
    comet_planet_ids, comet_path_index = _comet_metadata(obs_tensors, device)
    x, y, alive_by_step = _apply_comet_paths(
        x=x,
        y=y,
        alive_by_step=alive_by_step,
        planet_ids=planet_ids,
        comet_planet_ids=comet_planet_ids,
        comet_path_index=comet_path_index,
        obs_tensors=obs_tensors,
        offsets=offsets_long,
    )
    # Override slots where offset == 0 with the obs frame (truth at "now").
    zero_idx = (offsets_long == 0).nonzero(as_tuple=True)[0]
    if int(zero_idx.numel()) > 0:
        x[zero_idx, :] = obs.x.to(dtype=dtype).view(1, P)
        y[zero_idx, :] = obs.y.to(dtype=dtype).view(1, P)
        alive_by_step[zero_idx, :] = obs.alive.view(1, P)
    return {
        "x": x,
        "y": y,
        "alive_by_step": alive_by_step,
        "planet_ids": planet_ids,
        "radii": radii,
        "owner": owner,
        "ships": ships,
        "prod": prod,
        "step": step,
        "comet_planet_ids": comet_planet_ids,
        "comet_path_index": comet_path_index,
        "_offsets": offsets_long,
    }
def _comet_metadata(obs_tensors: dict, device: torch.device) -> tuple[Tensor, Tensor]:
    comets = obs_tensors.get("comets") or {}
    comet_ids = comets.get("planet_ids")
    if comet_ids is None:
        flat_ids = obs_tensors.get("comet_planet_ids")
        if flat_ids is None:
            flat_ids = torch.full((0,), -1, dtype=torch.long, device=device)
        else:
            flat_ids = flat_ids.to(device=device, dtype=torch.long)
        path_index = torch.full((0,), -1, dtype=torch.long, device=device)
        return flat_ids, path_index
    comet_ids = comet_ids.to(device=device, dtype=torch.long)
    flat_ids = comet_ids.reshape(-1)
    path_index = comets.get("path_index")
    if path_index is None:
        path_index = torch.full((comet_ids.shape[0],), -1, dtype=torch.long, device=device)
    else:
        path_index = path_index.to(device=device, dtype=torch.long)
    return flat_ids, path_index
def _apply_comet_paths(
    *,
    x: Tensor,
    y: Tensor,
    alive_by_step: Tensor,
    planet_ids: Tensor,
    comet_planet_ids: Tensor,
    comet_path_index: Tensor,
    obs_tensors: dict,
    offsets: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply comet path overrides at the requested integer step ``offsets``.
    ``x``/``y``/``alive_by_step`` are shaped ``[M, P]`` where ``M ==
    offsets.shape[0]``. The offsets tensor is 1D long.
    """
    comets = obs_tensors.get("comets") or {}
    paths = comets.get("paths")
    ids_grid = comets.get("planet_ids")
    if paths is None or ids_grid is None or comet_planet_ids.numel() == 0:
        return x, y, alive_by_step
    M, P = x.shape
    paths = paths.to(device=x.device, dtype=x.dtype)            # [E, C, T, 2]
    ids_grid = ids_grid.to(device=x.device, dtype=torch.long)   # [E, C]
    E = int(ids_grid.shape[0])
    C = int(ids_grid.shape[1])
    T = int(paths.shape[2])
    if E == 0 or C == 0 or T == 0:
        return x, y, alive_by_step
    flat_ids = ids_grid.reshape(E * C)                          # [E*C]
    matches = (planet_ids.unsqueeze(1) == flat_ids.unsqueeze(0)) & (flat_ids.unsqueeze(0) >= 0)  # [P, E*C]
    is_comet = matches.any(dim=1)                               # [P]
    flat_slot = matches.to(torch.float32).argmax(dim=1).long()  # [P]
    flat_paths_x = paths[..., 0].reshape(E * C, T)              # [E*C, T]
    flat_paths_y = paths[..., 1].reshape(E * C, T)
    path_x_by_slot = flat_paths_x[flat_slot]                    # [P, T]
    path_y_by_slot = flat_paths_y[flat_slot]
    finite = torch.isfinite(flat_paths_x)                       # [E*C, T]
    path_len = finite.sum(dim=1).to(dtype=torch.long)           # [E*C]
    len_by_slot = path_len[flat_slot]                           # [P]
    group_idx = (flat_slot // C).clamp(min=0, max=max(E - 1, 0))  # [P]
    path_idx_by_slot = comet_path_index[group_idx]             # [P]
    offsets_v = offsets.to(device=x.device, dtype=torch.long).view(M, 1)   # [M, 1]
    future_idx = path_idx_by_slot.view(1, P) + offsets_v        # [M, P]
    valid_future = (
        is_comet.view(1, P)
        & (future_idx >= 0)
        & (future_idx < len_by_slot.view(1, P))
    )                                                          # [M, P]
    idx_clamped = future_idx.clamp(min=0, max=max(T - 1, 0))    # [M, P]
    p_index = torch.arange(P, device=x.device).view(1, P).expand(M, P)
    comet_x = path_x_by_slot[p_index, idx_clamped]             # [M, P]
    comet_y = path_y_by_slot[p_index, idx_clamped]
    x = torch.where(valid_future, comet_x, x)
    y = torch.where(valid_future, comet_y, y)
    alive_by_step = torch.where(is_comet.view(1, P), valid_future, alive_by_step)
    return x, y, alive_by_step
def _same_2d(a: Tensor, b: Tensor) -> bool:
    if a.shape != b.shape:
        return False
    if a.numel() == 0:
        return True
    return bool((a == b.to(device=a.device, dtype=a.dtype)).all())
def _position_matches(
    pred_x: Tensor,
    pred_y: Tensor,
    cur_x: Tensor,
    cur_y: Tensor,
    alive: Tensor,
    epsilon: float,
) -> bool:
    diff = torch.maximum((pred_x - cur_x).abs(), (pred_y - cur_y).abs())
    diff = torch.where(alive, diff, torch.zeros_like(diff))
    return bool((diff <= float(epsilon)).all())
# ==========================================================================
# orbit_lite.garrison_launch
# ==========================================================================
"""What-if-I-launch flow projection over a :class:`PlanetGarrisonStatus`.
``PlanetGarrisonStatus`` is a per-planet ledger of projected owner / ships over a
future horizon, computed from the fleets we currently know about, assuming we do
nothing. :func:`sparse_launch_flow_delta` answers the forward-looking question an
agent faces — *"if I launch these ships, how does each player\'s net ship flow
(production minus combat losses) change?"* — by recomputing the production→combat
recurrence only for the planets a launch touches and diffing against the baseline.
A launch is two-sided: it debits the source planet's garrison (ships leave now,
before that turn's production) and credits the target's arrival at step ``k``.
Two leading axes are supported (single game):
- ``C`` — candidates: the different launches / launch-sets being scored.
- ``L`` — launches within a candidate: a candidate *is* a set of launches; ``L``
  is summed away during aggregation and is not an output axis.
Pass launches as ``[L]`` (no candidate axis) or ``[C, L]``.
"""
from dataclasses import dataclass
import torch
from torch import Tensor
@dataclass(frozen=True)
class LaunchSet:
    """A batched set of hypothetical launches issued on the current turn.
    All tensors share a leading prefix (empty or ``[C]``) followed by a
    trailing launch axis ``L`` (use ``L=1`` for a single launch). ``eta`` is in
    steps from the current frame (arrival lands at garrison step ``k = eta``;
    ``eta`` must be ``>= 1``). ``owner`` defaults to the acting player but is
    per-launch so opponent what-ifs are expressible.
    """
    source_slots: Tensor  # [*prefix, L] long  (planet slot to launch FROM)
    target_slots: Tensor  # [*prefix, L] long  (planet slot to launch TO)
    ships: Tensor         # [*prefix, L] float
    eta: Tensor           # [*prefix, L] float/long (steps to arrival, >= 1)
    owner: Tensor         # [*prefix, L] long
    valid: Tensor         # [*prefix, L] bool
    @property
    def has_candidate_axis(self) -> bool:
        return self.source_slots.dim() >= 2
def _per_step_survivor(arrivals: Tensor) -> tuple[Tensor, Tensor]:
    """Engine survivor over the owner axis for every step.
    ``arrivals`` is ``[..., A]``; returns ``(survivor_owner, survivor_ships)``
    over the trailing axis, applying the engine rule: survivor ships = top1 -
    top2, ties annihilate (ships 0). Owner is meaningful only where ships > 0.
    """
    A = int(arrivals.shape[-1])
    if A >= 2:
        top2 = arrivals.topk(k=2, dim=-1)
        top_ships = top2.values[..., 0]
        second_ships = top2.values[..., 1]
        top_owner = top2.indices[..., 0].to(dtype=torch.long)
    else:
        top_ships, top_owner = arrivals.max(dim=-1)
        second_ships = torch.zeros_like(top_ships)
        top_owner = top_owner.to(dtype=torch.long)
    tied = top_ships == second_ships
    survivor_ships = torch.where(
        tied, torch.zeros_like(top_ships), (top_ships - second_ships).clamp(min=0.0)
    )
    return top_owner, survivor_ships
def _run_exact_recurrence(
    *,
    init_owner: Tensor,   # [N, P] long
    init_ships: Tensor,   # [N, P] float (already source-debited)
    prod: Tensor,         # [N, P] float
    alive: Tensor,        # [N, P, H+1] bool
    arrivals: Tensor,     # [N, P, H, A] float (steps 1..H, baseline + delta)
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Walk the engine production→combat recurrence over ``k = 1..H``.
    Mirrors ``PlanetMovement._fill_garrison_trajectory`` Half B exactly, but for
    all planets without the simple/complex fast-path split (clarity over the few
    saved kernels — this is the reference path). Returns
    ``(owner, ships, pre_owner, pre_ships)`` each ``[N, P, H+1]`` with step 0 set
    to the recurrence's starting state.
    """
    N, P = init_owner.shape
    H = int(arrivals.shape[2])
    device = init_ships.device
    owner_out = torch.empty(N, P, H + 1, dtype=init_owner.dtype, device=device)
    ships_out = torch.empty(N, P, H + 1, dtype=init_ships.dtype, device=device)
    pre_owner_out = torch.empty_like(owner_out)
    pre_ships_out = torch.empty_like(ships_out)
    owner_out[..., 0] = init_owner
    ships_out[..., 0] = init_ships
    pre_owner_out[..., 0] = init_owner
    pre_ships_out[..., 0] = init_ships
    survivor_owner, survivor_ships = _per_step_survivor(arrivals)  # [N, P, H]
    state_owner = init_owner.clone()
    state_ships = init_ships.clone()
    zero_ships = torch.zeros((), dtype=state_ships.dtype, device=device)
    neg_one = torch.full((), -1, dtype=state_owner.dtype, device=device)
    zero_prod = torch.zeros((), dtype=prod.dtype, device=device)
    for k in range(1, H + 1):
        a_before = alive[..., k - 1]
        a_now = alive[..., k]
        s_owner = survivor_owner[..., k - 1]
        s_ships = survivor_ships[..., k - 1]
        # Production: owned planets alive at the start of this step.
        produces = a_before & (state_owner >= 0)
        state_ships = state_ships + torch.where(produces, prod, zero_prod)
        # Pre-combat snapshot (after production, before same-step combat).
        pre_owner_out[..., k] = torch.where(a_now, state_owner, neg_one)
        pre_ships_out[..., k] = torch.where(a_now, state_ships, zero_ships)
        # Survivor vs the prior garrison.
        has_combat = (s_ships > 0.0) & a_now
        same = state_owner == s_owner
        diff = state_ships - s_ships
        attacker_wins = (~same) & (diff < 0.0)
        combat_ships = torch.where(same, state_ships + s_ships, diff.abs())
        combat_owner = torch.where(attacker_wins, s_owner, state_owner)
        state_ships = torch.where(has_combat, combat_ships, state_ships)
        state_owner = torch.where(has_combat, combat_owner, state_owner)
        # End-of-step death reset.
        state_owner = torch.where(a_now, state_owner, neg_one)
        state_ships = torch.where(a_now, state_ships, zero_ships)
        owner_out[..., k] = state_owner
        ships_out[..., k] = state_ships
    return owner_out, ships_out, pre_owner_out, pre_ships_out
def _validate_inputs(
    status: PlanetGarrisonStatus,
    prod: Tensor,
    alive_by_step: Tensor,
    player_count: int,
) -> tuple[int, int, int, int]:
    """Check shapes and return ``(B, P, H, A)``."""
    if status.arrivals_by_owner is None:
        raise ValueError(
            "garrison status must carry arrivals_by_owner (build it from a "
            "PlanetMovement with track_fleets=True)"
        )
    if status.pre_combat_owner is None or status.pre_combat_ships is None:
        raise ValueError("garrison status must carry pre_combat_owner/ships")
    if status.owner.dim() != 2:
        raise ValueError(
            "expected a full-board status with owner shaped [P, H+1]; got "
            f"{tuple(status.owner.shape)}"
        )
    P, H1 = status.owner.shape
    H = H1 - 1
    A = int(status.arrivals_by_owner.shape[-1])
    if int(player_count) != A:
        raise ValueError(
            f"player_count={player_count} disagrees with arrivals owner axis A={A}"
        )
    if tuple(prod.shape) != (P,):
        raise ValueError(f"prod must be [P]=({P},); got {tuple(prod.shape)}")
    if tuple(alive_by_step.shape) != (H1, P):
        raise ValueError(
            f"alive_by_step must be [H+1, P]=({H1}, {P}); got "
            f"{tuple(alive_by_step.shape)}"
        )
    return P, H, A
# ---------------------------------------------------------------------------
# Per-player flow accounting: diff two garrison statuses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GarrisonFlowDiff:
    """Difference in per-player flow between a current and a hypothetical status.
    Each field is ``[*prefix, A]`` (per player). ``*_delta`` is
    ``hypothetical - current``. ``net_ship_delta`` is the change in net ships
    gained (``produced - lost_to_combat``) — i.e. how much better/worse off each
    player ends up under the hypothetical, ignoring ships in transit.
    """
    player_id: int
    ships_produced_current: Tensor
    ships_produced_hypothetical: Tensor
    ships_produced_delta: Tensor
    ships_lost_combat_current: Tensor
    ships_lost_combat_hypothetical: Tensor
    ships_lost_combat_delta: Tensor
    net_ship_delta: Tensor
    @property
    def player_count(self) -> int:
        return int(self.ships_produced_delta.shape[-1])
def _flow_terms_per_planet(
    *,
    owner: Tensor,        # [.., P, H+1]
    pre_owner: Tensor,    # [.., P, H+1]
    pre_ships: Tensor,    # [.., P, H+1]
    arr_full: Tensor,     # [.., P, H+1, A]
    prod: Tensor,         # [.., P] (broadcastable)
    alive_pmajor: Tensor, # [.., P, H+1] (broadcastable, planet-major)
) -> tuple[Tensor, Tensor]:
    """Per-planet production and combat losses, summed over the horizon only.
    Returns ``(produced, combat_lost)`` each ``[.., P, A]``. Combat follows
    the engine combat rule; production credits ``prod`` to the owner
    holding the planet entering each step (from ``prod``, not ship deltas, so a
    source launch debit is not mistaken for negative production).
    """
    A = int(arr_full.shape[-1])
    H = int(owner.shape[-1]) - 1
    fdtype = pre_ships.dtype
    a_idx = torch.arange(A, device=owner.device)
    # Production credited to owner at the start of each step (= owner[k-1]).
    producing_owner = owner[..., :H]                                 # [.., P, H]
    amount = prod.unsqueeze(-1) * alive_pmajor[..., :H].to(fdtype)   # [.., P, H]
    prod_owner_oh = producing_owner.unsqueeze(-1) == a_idx           # [.., P, H, A]
    produced = (amount.unsqueeze(-1) * prod_owner_oh.to(fdtype)).sum(dim=-2)  # [.., P, A]
    # Combat per step (engine top1 - top2 survivor, then survivor vs garrison).
    arr_k = arr_full[..., 1:, :]
    survivor_owner, survivor_ships = _per_step_survivor(arr_k)       # [.., P, H]
    survived = torch.where(
        a_idx == survivor_owner.unsqueeze(-1),
        survivor_ships.unsqueeze(-1),
        torch.zeros_like(survivor_ships).unsqueeze(-1),
    )
    attacker_lost = (arr_k - survived).clamp(min=0.0)                # [.., P, H, A]
    prior_owner = pre_owner[..., 1:]
    prior_ships = pre_ships[..., 1:]
    fights_garrison = (survivor_ships > 0.0) & (survivor_owner != prior_owner) & (survivor_owner >= 0)
    garrison_loss = torch.where(
        fights_garrison, torch.minimum(prior_ships, survivor_ships), torch.zeros_like(prior_ships)
    )
    is_survivor = (a_idx == survivor_owner.unsqueeze(-1)) & fights_garrison.unsqueeze(-1)
    is_prior = (
        (a_idx == prior_owner.unsqueeze(-1))
        & fights_garrison.unsqueeze(-1)
        & (prior_owner >= 0).unsqueeze(-1)
    )
    garrison_lost = garrison_loss.unsqueeze(-1) * (is_survivor.to(fdtype) + is_prior.to(fdtype))
    combat_lost = (attacker_lost + garrison_lost).sum(dim=-2)        # [.., P, A]
    return produced, combat_lost
# ---------------------------------------------------------------------------
# Sparse prototype: per-candidate flow deltas without dense [C, P, H, A]
# ---------------------------------------------------------------------------
def _normalize_launches_bcl(launches: LaunchSet) -> tuple[Tensor, ...]:
    """Return ``(src, tgt, ships, eta, owner, valid)`` shaped ``[C, L]``."""
    fields = (
        launches.source_slots, launches.target_slots, launches.ships,
        launches.eta, launches.owner, launches.valid,
    )
    if launches.has_candidate_axis:
        return fields
    return tuple(f.unsqueeze(0) for f in fields)  # [L] -> [1, L]
def sparse_launch_flow_delta(
    status: PlanetGarrisonStatus,
    *,
    prod: Tensor,
    alive_by_step: Tensor,
    player_count: int,
    launches: LaunchSet,
    player_id: int = 0,
) -> GarrisonFlowDiff:
    """Sparse equivalent of ``diff_garrison_flow(status, apply_launches_exact(...))``.
    Returns the **same** exact per-candidate, per-player flow diff as the dense
    pipeline, but never materializes the dense ``[C, P, H, A]`` arrivals or a
    ``[C, P, H+1]`` trajectory. It exploits two facts:
    - the garrison projection is per-planet independent given the arrival
      buckets, so a launch only changes the trajectory of the planets it touches
      (its source, via the debit, and its target, via the credit);
    - untouched planets contribute zero to the flow *delta*.
    So it recomputes the recurrence only for the affected ``(candidate, planet)``
    cells (~2 per candidate for single launches, vs all ``P``) and scatter-adds
    their per-planet flow deltas into ``[C, A]``. Cost and memory scale with
    the number of affected cells, not ``B·C·P``.
    """
    P, H, A = _validate_inputs(status, prod, alive_by_step, player_count)
    device = status.owner.device
    fdtype = status.ships.dtype
    assert status.pre_combat_owner is not None and status.pre_combat_ships is not None
    assert status.arrivals_by_owner is not None
    src, tgt, ships, eta, owner, valid = _normalize_launches_bcl(launches)
    C = int(src.shape[0])
    L = int(src.shape[-1])
    src = src.to(device=device, dtype=torch.long)
    tgt = tgt.to(device=device, dtype=torch.long)
    ships = ships.to(device=device, dtype=fdtype)
    owner = owner.to(device=device, dtype=torch.long)
    valid = valid.to(device=device, dtype=torch.bool)
    h_idx = torch.ceil(eta.to(device=device, dtype=fdtype)).to(torch.long) - 1
    valid_t = valid & (ships > 0) & (tgt >= 0) & (tgt < P) & (owner >= 0) & (owner < A) & (h_idx >= 0) & (h_idx < H)
    valid_s = valid & (ships > 0) & (src >= 0) & (src < P)
    src_safe = src.clamp(0, max(P - 1, 0))
    tgt_safe = tgt.clamp(0, max(P - 1, 0))
    # Affected planets per candidate (source debit OR target credit). [C, P]
    affected = torch.zeros(C, P, dtype=fdtype, device=device)
    affected.scatter_add_(1, src_safe, valid_s.to(fdtype))
    affected.scatter_add_(1, tgt_safe, valid_t.to(fdtype))
    affected_mask = affected > 0
    # Baseline per-planet flow (shared across candidates).
    base_prod_pp, base_combat_pp = _flow_terms_per_planet(
        owner=status.owner,
        pre_owner=status.pre_combat_owner,
        pre_ships=status.pre_combat_ships,
        arr_full=status.arrivals_by_owner,
        prod=prod,
        alive_pmajor=alive_by_step.permute(1, 0),
    )                                                        # [P, A]
    base_prod = base_prod_pp.sum(dim=0)                      # [A]
    base_combat = base_combat_pp.sum(dim=0)
    produced_delta = torch.zeros(C, A, dtype=fdtype, device=device)
    combat_delta = torch.zeros(C, A, dtype=fdtype, device=device)
    if bool(affected_mask.any()):
        c_aff, p_aff = affected_mask.nonzero(as_tuple=True)         # [N]
        N = int(c_aff.numel())
        cell_id = torch.full((C, P), -1, dtype=torch.long, device=device)
        cell_id[c_aff, p_aff] = torch.arange(N, device=device)
        # Source debit per affected cell.
        debit_cp = torch.zeros(C, P, dtype=fdtype, device=device)
        debit_cp.scatter_add_(1, src_safe, torch.where(valid_s, ships, torch.zeros_like(ships)))
        debit_aff = debit_cp[c_aff, p_aff]                          # [N]
        # Target credits scattered onto the affected cells: [N, H, A].
        arr_aff = torch.zeros(N, H, A, dtype=fdtype, device=device)
        launch_cell = cell_id.gather(1, tgt_safe)                   # [C, L]
        m = valid_t
        cells, hh, oo, ss = launch_cell[m], h_idx[m], owner[m], ships[m]
        ok = cells >= 0
        arr_aff.index_put_((cells[ok], hh[ok], oo[ok]), ss[ok], accumulate=True)
        base_arr_k = status.arrivals_by_owner[..., 1:, :]           # [P, H, A]
        arrivals_cell = base_arr_k[p_aff] + arr_aff                 # [N, H, A]
        init_owner = status.owner[p_aff, 0]                         # [N]
        init_ships = (status.ships[p_aff, 0] - debit_aff).clamp(min=0.0)
        prod_aff = prod[p_aff]                                      # [N]
        alive_aff = alive_by_step[:, p_aff].transpose(0, 1)         # [N, H+1]
        # One-planet recurrence per affected cell (P=1 lane).
        o_t, _s_t, po_t, ps_t = _run_exact_recurrence(
            init_owner=init_owner.unsqueeze(1),
            init_ships=init_ships.unsqueeze(1),
            prod=prod_aff.unsqueeze(1),
            alive=alive_aff.unsqueeze(1),
            arrivals=arrivals_cell.unsqueeze(1),
        )
        zero_frame = torch.zeros(N, 1, 1, A, dtype=fdtype, device=device)
        arr_full_cell = torch.cat([zero_frame, arrivals_cell.unsqueeze(1)], dim=-2)
        hyp_prod_pp, hyp_combat_pp = _flow_terms_per_planet(
            owner=o_t, pre_owner=po_t, pre_ships=ps_t, arr_full=arr_full_cell,
            prod=prod_aff.unsqueeze(1), alive_pmajor=alive_aff.unsqueeze(1),
        )
        dprod = hyp_prod_pp.squeeze(1) - base_prod_pp[p_aff]            # [N, A]
        dcombat = hyp_combat_pp.squeeze(1) - base_combat_pp[p_aff]
        produced_delta.index_put_((c_aff,), dprod, accumulate=True)
        combat_delta.index_put_((c_aff,), dcombat, accumulate=True)
    produced_current = base_prod.unsqueeze(0)                      # [1, A]
    combat_current = base_combat.unsqueeze(0)
    diff = GarrisonFlowDiff(
        player_id=int(player_id),
        ships_produced_current=produced_current,
        ships_produced_hypothetical=produced_current + produced_delta,
        ships_produced_delta=produced_delta,
        ships_lost_combat_current=combat_current,
        ships_lost_combat_hypothetical=combat_current + combat_delta,
        ships_lost_combat_delta=combat_delta,
        net_ship_delta=produced_delta - combat_delta,
    )
    # Squeeze the candidate axis back out for [L] launches (C == 1, no axis).
    if not launches.has_candidate_axis:
        def _sq(t: Tensor) -> Tensor:
            return t.squeeze(0)
        diff = GarrisonFlowDiff(
            player_id=diff.player_id,
            ships_produced_current=base_prod,
            ships_produced_hypothetical=_sq(diff.ships_produced_hypothetical),
            ships_produced_delta=_sq(diff.ships_produced_delta),
            ships_lost_combat_current=base_combat,
            ships_lost_combat_hypothetical=_sq(diff.ships_lost_combat_hypothetical),
            ships_lost_combat_delta=_sq(diff.ships_lost_combat_delta),
            net_ship_delta=_sq(diff.net_ship_delta),
        )
    return diff
# ==========================================================================
# orbit_lite.distance_cache
# ==========================================================================
"""Cross-k distance cache for the movement-backed planner.
Entry ``cross_dist[k, s, t]`` is the Euclidean distance from planet ``s`` at step
0 to planet ``t`` at step ``k`` — the *cross-time* distance a fleet must travel if
it launches now from ``s`` to intercept ``t`` at time ``k``. For static planets
this equals same-step pairwise distance; for orbiting sources the cross-time form
is the geometrically correct quantity for fleet-intercept feasibility. A
precomputed ``[K+1, P, P]`` window gives exact per-step lookups for free.
"""
from dataclasses import dataclass
import torch
from torch import Tensor
@dataclass
class DistanceCache:
    """Per-turn cross-k distance window.
    Tensor shapes:
    - ``cross_dist``: ``[K+1, P, P]`` -- ``[k, s, t] = dist(s@0, t@k)``.
    - ``alive_by_step``: ``[K+1, P]`` -- view sliced from
      ``movement.alive_by_step``.
    """
    cross_dist: Tensor
    alive_by_step: Tensor
    K: int
    @property
    def P(self) -> int:
        return int(self.cross_dist.shape[-1])
    @property
    def device(self) -> torch.device:
        return self.cross_dist.device
    @property
    def dtype(self) -> torch.dtype:
        return self.cross_dist.dtype
def build_distance_cache(
    movement: PlanetMovement,
    *,
    max_k: int,
) -> DistanceCache:
    """Build a fresh cross-k distance cache from the rolling movement cache.
    ``max_k`` is clamped to ``movement.movement_horizon``. Caller is
    expected to clamp its own k queries the same way.
    """
    K = max(0, min(int(max_k), int(movement.movement_horizon)))
    P = int(movement.P)
    src_x0 = movement.x[0]                         # [P]
    src_y0 = movement.y[0]
    tgt_x = movement.x[: K + 1]                    # [K+1, P]
    tgt_y = movement.y[: K + 1]
    # cross[k, s, t] = dist(s@0, t@k)
    dx = src_x0.view(1, P, 1) - tgt_x.unsqueeze(1)
    dy = src_y0.view(1, P, 1) - tgt_y.unsqueeze(1)
    cross_dist = torch.sqrt((dx * dx + dy * dy).clamp(min=0.0))
    alive_by_step = movement.alive_by_step[: K + 1]
    return DistanceCache(
        cross_dist=cross_dist,
        alive_by_step=alive_by_step,
        K=K,
    )
# ---------------------------------------------------------------------------
# Min-distance helper (replaces movement_min_distance_to_targets)
# ---------------------------------------------------------------------------
def min_distance_to_targets(
    cache: DistanceCache,
    source_mask: Tensor,
    target_mask: Tensor,
    *,
    max_k: int,
) -> Tensor:
    """Return per-target nearest-source distance using cross-k lookups.
    For each target ``t``, return the smallest
    ``dist(s@0, t@k)`` over alive valid sources ``s`` and steps
    ``k in [1, min(max_k, cache.K)]``. This is the exact analogue of
    ``movement_min_distance_to_targets`` with sampled steps replaced by the
    full integer range.
    """
    if source_mask.shape[-1] != cache.P or target_mask.shape[-1] != cache.P:
        raise ValueError("source_mask and target_mask must have shape [P]")
    K = max(0, min(int(max_k), int(cache.K)))
    if K <= 0:
        return torch.zeros(cache.P, dtype=cache.dtype, device=cache.device)
    # Clone the cross-k slice so we can ``masked_fill_`` invalid entries to +inf
    # without touching the cache's storage. The union of the three masks is
    # equivalent to ``~valid_pair = ~src_mask | ~tgt_mask | ~alive_at_k``.
    cross = cache.cross_dist[1 : K + 1].clone()    # [K, P_src, P_tgt]
    alive_steps = cache.alive_by_step[1 : K + 1]   # [K, P]
    src_mask = source_mask.to(device=cache.device, dtype=torch.bool)
    tgt_mask = target_mask.to(device=cache.device, dtype=torch.bool)
    inf_v = float("inf")
    cross.masked_fill_(~alive_steps.unsqueeze(1), inf_v)
    cross.masked_fill_(~src_mask.view(1, cache.P, 1), inf_v)
    cross.masked_fill_(~tgt_mask.view(1, 1, cache.P), inf_v)
    best_per_target = cross.amin(dim=(0, 1))       # over K and source axis
    return torch.where(torch.isfinite(best_per_target), best_per_target, torch.zeros_like(best_per_target))
# ---------------------------------------------------------------------------
# Compact candidate pairs (replaces compact_candidate_pairs for regroup)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Aiming reachability mask (precheck augmentation for movement_pairwise_grid)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ==========================================================================
# orbit_lite.intercept_aim
# ==========================================================================
"""Fixed-fleet intercept aim — sub-turn-accurate angle for an orbiting target.
Solves the **continuous** intercept time ``t*`` (root of
``v·t = dist(target_pos(t), source) − gap`` with the target on its analytic
orbit), aims at ``target_pos(t*)``, and verifies that angle with a
fully-vectorized analytic first-contact check.
* **Root** — a continuous fixed-point iteration (no grid scan / argmax /
  bisection), free of grid-resolution artifacts.
* **Verify** — :func:`_analytic_first_contact` reproduces the engine's
  first-contact verdict exactly (swept-pair vs every planet, sun, bounds,
  lowest-slot same-step tie-break) with no engine state and no per-step loop.
  A shot is viable iff it contacts the target first.
Returns ``angle`` / ``eta`` / ``viable``.
"""
import torch
from torch import Tensor
_FP_ITERS = 6  # continuous fixed-point iterations for the intercept time
_BIG = 1_000_000.0
def intercept_angle(
    movement: PlanetMovement,
    source_slots: Tensor,
    target_slots: Tensor,
    fleet_sizes: Tensor,
    *,
    fp_iters: int = _FP_ITERS,
    active: Tensor | None = None,
) -> dict[str, Tensor]:
    """Continuous-intercept aim for a fixed fleet size (the root angle only).
    Broadcastable slot/size tensors in; ``{angle, eta, viable}`` out (same shape).
    Non-viable candidates get ``eta == inf``.
    ``active`` (optional, broadcastable to the candidate shape): a reachability
    precheck that gates the expensive body screen. The lead angle is still solved
    on the full grid, so kept candidates' angles are bit-identical; only the
    integer-exact first-contact screen is compacted to the active candidates.
    Candidates with ``active`` False resolve to non-viable. Pass a strict superset
    of viability (e.g. :func:`planner_core.reachable_mask`) for a zero-behaviour-change
    speedup — ``None`` screens everything.
    """
    dev = movement.device
    dt = movement.dtype
    H = int(movement.movement_horizon)
    src, tgt, ships = torch.broadcast_tensors(
        source_slots.to(device=dev),
        target_slots.to(device=dev),
        fleet_sizes.to(device=dev, dtype=dt),
    )
    shape = src.shape
    src = src.long().clamp(0, max(movement.P - 1, 0)).reshape(-1)
    tgt = tgt.long().clamp(0, max(movement.P - 1, 0)).reshape(-1)
    ships = ships.to(dt).clamp(min=1.0).reshape(-1)
    M = src.shape[0]
    sx, sy = movement.position_at_slots(src, 0)                       # [M]
    src_r = movement.radii[src]
    tgt_r = movement.radii[tgt]
    speed = fleet_speed(ships).clamp(min=1e-6)                        # [M]
    # Target orbit params from its integer positions: centre-relative radius +
    # phase at t=0 and the per-step angular step (auto-zero for static planets).
    t0x, t0y = movement.position_at_slots(tgt, 0)
    t1x, t1y = movement.position_at_slots(tgt, 1)
    R = torch.sqrt(((t0x - CENTER) ** 2 + (t0y - CENTER) ** 2).clamp(min=0.0))
    a0 = torch.atan2(t0y - CENTER, t0x - CENTER)
    a1 = torch.atan2(t1y - CENTER, t1x - CENTER)
    omega = torch.atan2(torch.sin(a1 - a0), torch.cos(a1 - a0))       # wrapped Δangle/step
    gap = src_r + LAUNCH_SURFACE_OFFSET + tgt_r + TARGET_HIT_SURFACE_OFFSET
    def target_pos(t: Tensor):
        ang = a0 + omega * t
        return CENTER + R * torch.cos(ang), CENTER + R * torch.sin(ang)
    # Continuous fixed point t = (dist(target_pos(t), src) - gap)/v, seeded with
    # the static-target estimate. A contraction whenever the target's radial speed
    # stays below the fleet speed (true for reachable shots); divergent guesses
    # just produce a bad angle that the verify rejects.
    d0 = torch.sqrt(((t0x - sx) ** 2 + (t0y - sy) ** 2).clamp(min=0.0))
    t_star = ((d0 - gap) / speed).clamp(min=0.0, max=float(H))
    for _ in range(int(fp_iters)):
        tx, ty = target_pos(t_star)
        d = torch.sqrt(((tx - sx) ** 2 + (ty - sy) ** 2).clamp(min=0.0))
        t_star = ((d - gap) / speed).clamp(min=0.0, max=float(H))
    tx, ty = target_pos(t_star)
    angle = torch.atan2(ty - sy, tx - sx)                             # [M]
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    launch_x = sx + cos_a * (src_r + LAUNCH_SURFACE_OFFSET)           # [M]
    launch_y = sy + sin_a * (src_r + LAUNCH_SURFACE_OFFSET)
    # Relevant flight length = distance to the intercept (+margins for the arrival
    # step and the target radius). Bounds the broad-phase cull segment to the
    # fleet's actual launch→target path. Planets beyond the target can never be the
    # first contact for a target-reaching fleet, so this preserves `viable`
    # (contact==tgt) and the viable-case `eta` exactly.
    eta_cap = (t_star + 2.0).clamp(max=float(H))
    seg_len = speed * eta_cap + tgt_r + 2.0                            # [M]
    px = movement.x[: H + 1, :]                                       # [H+1, P] (already cached)
    py = movement.y[: H + 1, :]
    radii_p = movement.radii
    alive0 = movement.alive_at(0)
    if active is None:
        contact, eta_c = _analytic_first_contact(
            launch_x=launch_x, launch_y=launch_y, cos_a=cos_a, sin_a=sin_a,
            speed=speed, px=px, py=py, p_alive0=alive0,
            radii=radii_p, H=H, seg_len=seg_len,
        )                                                             # [M]
    else:
        # Reachability gate: screen only the active candidates. The per-candidate
        # integer contact/eta are shortlist-independent, so kept candidates' verdicts
        # are bit-identical to the full screen. Compact to the active candidates,
        # screen, then scatter home; inactive cells resolve to contact = -1.
        act = active.broadcast_to(shape).reshape(M).to(torch.bool)
        n_max = max(1, int(act.sum().item()))
        order = (~act).to(torch.int8).argsort(stable=True)           # active cells first
        midx = order[:n_max]                                         # [n_max]
        keep = act[midx]
        contact_m, eta_cm = _analytic_first_contact(
            launch_x=launch_x[midx], launch_y=launch_y[midx],
            cos_a=cos_a[midx], sin_a=sin_a[midx],
            speed=speed[midx], px=px, py=py, p_alive0=alive0,
            radii=radii_p, H=H, seg_len=seg_len[midx],
        )                                                            # [n_max]
        contact = torch.full((M,), -1, dtype=contact_m.dtype, device=dev)
        eta_c = torch.full((M,), float(H), dtype=eta_cm.dtype, device=dev)
        contact[midx] = torch.where(keep, contact_m, torch.full_like(contact_m, -1))
        eta_c[midx] = torch.where(keep, eta_cm, torch.full_like(eta_cm, float(H)))
    viable = contact == tgt                                           # [M]
    eta_out = torch.where(viable, eta_c.to(dt), torch.full_like(eta_c.to(dt), float("inf")))
    return {
        "angle": angle.reshape(shape),
        "eta": eta_out.reshape(shape),
        "viable": viable.reshape(shape),
    }
def _analytic_first_contact(
    *,
    launch_x: Tensor,
    launch_y: Tensor,
    cos_a: Tensor,
    sin_a: Tensor,
    speed: Tensor,
    px: Tensor,
    py: Tensor,
    p_alive0: Tensor,
    radii: Tensor,
    H: int,
    seg_len: Tensor | None = None,
    max_bytes: int = 256 * 1024 * 1024,
):
    """First planet a fleet contacts, engine-faithful, shaped ``[M, C]``.
    Reproduces batch ``_move_fleets`` exactly: straight fleet motion at ``speed``,
    swept-pair collision vs every step-0-alive planet, OOB + point-to-segment sun
    kill (only when no planet was hit that step), and the lowest-slot same-step
    tie-break. ``launch_*``/``cos_a``/``sin_a``/``speed`` are ``[M]``; ``px``,
    ``py`` are ``[H+1, P]`` planet positions per step; ``p_alive0`` is ``[P]``
    (step-0 alive); ``radii`` is ``[P]``.
    Returns ``(contact_slot, eta)`` — ``contact_slot == -1`` and ``eta == H`` when
    the fleet contacts no planet (or dies first).
    Two-phase to keep the exact swept-pair off the common clear-shot path:
    * **Broad phase** — an AABB cull (the fleet's full-horizon segment box vs each
      planet's swept box inflated by its radius). A planet whose box can't overlap
      the segment can never be hit, so it's dropped. The per-candidate shortlist
      collapses to the few real near-path planets (~1-3 for a clear shot vs ``P``).
      Conservative → the kept set always contains every hittable planet, so the
      result is **byte-identical** to checking all ``P``.
    * **Narrow phase** — the exact swept-pair only on the shortlisted planets,
      flattened to ``N = M`` candidates and run in byte-budgeted chunks (the
      dense ``[N,K,H]`` form would OOM when the regroup grid makes ``M`` large).
    ``amin`` reductions are order-independent so chunking/culling don't perturb the
    values (byte-exact + CPU≡CUDA guarantees hold). Runs eager; the one host sync
    (max shortlist length) is cheap.
    """
    M = cos_a.shape[0]
    P = px.shape[-1]
    dev = cos_a.device
    dt = launch_x.dtype
    N = M
    big = _BIG
    lx = launch_x.reshape(N); ly = launch_y.reshape(N)
    ca = cos_a.reshape(N); sa = sin_a.reshape(N); sp = speed.reshape(N)
    # --- Broad phase: AABB cull (no time axis → cheap). The conservative segment box
    # runs launch → launch + u·seg_len, where seg_len bounds the fleet's relevant
    # flight (distance to the intercept; falls back to the full horizon v·H). The
    # planet box is its swept extent over [0,H] inflated by its radius. ---
    slen = (sp * float(H)) if seg_len is None else seg_len.reshape(N)
    end_x = lx + ca * slen; end_y = ly + sa * slen
    seg_xmin = torch.minimum(lx, end_x); seg_xmax = torch.maximum(lx, end_x)   # [N]
    seg_ymin = torch.minimum(ly, end_y); seg_ymax = torch.maximum(ly, end_y)
    bb_xmin = px.amin(0) - radii                                              # [P]
    bb_xmax = px.amax(0) + radii
    bb_ymin = py.amin(0) - radii
    bb_ymax = py.amax(0) + radii
    keep = ~(
        (seg_xmax.unsqueeze(1) < bb_xmin) | (seg_xmin.unsqueeze(1) > bb_xmax)
        | (seg_ymax.unsqueeze(1) < bb_ymin) | (seg_ymin.unsqueeze(1) > bb_ymax)
    )                                                                          # [N, P]
    K = max(1, int(keep.sum(1).amax().item()))            # one host sync (eager-cheap)
    order = (~keep).to(torch.int8).argsort(dim=1, stable=True)                 # kept first
    shortlist = order[:, :K]                                                   # [N, K]
    valid = keep.gather(1, shortlist)                                          # [N, K]
    k = torch.arange(H + 1, device=dev, dtype=dt)                              # [H+1]
    t_ax = torch.arange(H + 1, device=dev).view(1, H + 1, 1)                   # [1,H+1,1]
    step_h = torch.arange(1, H + 1, device=dev, dtype=dt).view(1, H, 1)        # [1,H,1]
    # ~16 float intermediates of [chunk, H, K] dominate; budget the largest tensor.
    bytes_per = max(1, 16 * H * K * 4)
    chunk = max(4096, max_bytes // bytes_per)
    chunk = min(chunk, max(N, 1))
    contacts: list[Tensor] = []
    etas: list[Tensor] = []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        sl = shortlist[s:e]                                                   # [n, K]
        fx = lx[s:e].view(-1, 1) + ca[s:e].view(-1, 1) * sp[s:e].view(-1, 1) * k   # [n, H+1]
        fy = ly[s:e].view(-1, 1) + sa[s:e].view(-1, 1) * sp[s:e].view(-1, 1) * k
        # advanced-index the K shortlisted planets directly → [n, H+1, K] (no [n,H+1,P])
        sl_e = sl.view(-1, 1, K)
        pxc = px[t_ax, sl_e]                                                  # [n, H+1, K]
        pyc = py[t_ax, sl_e]
        radc = radii[sl]                                                      # [n, K]
        alivec = p_alive0[sl] & valid[s:e]                                    # [n, K]
        real_slot = sl.to(dt)                                                 # [n, K]
        fx0 = fx[:, :-1].unsqueeze(-1); fy0 = fy[:, :-1].unsqueeze(-1)        # [n,H,1]
        fx1 = fx[:, 1:].unsqueeze(-1);  fy1 = fy[:, 1:].unsqueeze(-1)
        hit = _swept_pair_hit_mask(
            fx0, fy0, fx1, fy1,
            pxc[:, :-1, :], pyc[:, :-1, :], pxc[:, 1:, :], pyc[:, 1:, :],
            radc.unsqueeze(1),
        )                                                                     # [n,H,K]
        hit = hit & alivec.unsqueeze(1)
        planet_hit_step = torch.where(hit, step_h, torch.full_like(step_h, big)).amin(1)  # [n,K]
        first_planet_step = planet_hit_step.amin(1)                           # [n]
        is_first = planet_hit_step == first_planet_step.unsqueeze(-1)
        contact_planet = torch.where(is_first, real_slot, torch.full_like(real_slot, big)).amin(1)  # [n]
        # env death: OOB at the new position OR the segment grazes the sun (static).
        nfx = fx[:, 1:]; nfy = fy[:, 1:]; ofx = fx[:, :-1]; ofy = fy[:, :-1]   # [n,H]
        oob = (nfx < 0) | (nfx > BOARD_SIZE) | (nfy < 0) | (nfy > BOARD_SIZE)
        vx = nfx - ofx; vy = nfy - ofy
        wx = CENTER - ofx; wy = CENTER - ofy
        vv = (vx * vx + vy * vy).clamp(min=1e-12)
        t = ((wx * vx + wy * vy) / vv).clamp(0.0, 1.0)
        cxp = ofx + t * vx; cyp = ofy + t * vy
        sun = ((cxp - CENTER) ** 2 + (cyp - CENTER) ** 2) < (SUN_RADIUS * SUN_RADIUS)
        env = oob | sun                                                       # [n,H]
        death_step = torch.where(env, step_h.squeeze(-1), torch.full_like(env, big, dtype=dt)).amin(1)  # [n]
        # Planet collision resolves BEFORE env removal in the same step (<=).
        ht = (first_planet_step <= death_step) & (first_planet_step < big)
        contacts.append(torch.where(ht, contact_planet, torch.full_like(contact_planet, -1.0)).long())
        etas.append(torch.where(ht, first_planet_step, torch.full_like(first_planet_step, float(H))))
    contact = (contacts[0] if len(contacts) == 1 else torch.cat(contacts)).view(M)
    eta = (etas[0] if len(etas) == 1 else torch.cat(etas)).view(M)
    return contact, eta
# ==========================================================================
# orbit_lite.movement_step
# ==========================================================================
from dataclasses import dataclass
from typing import Sequence
import torch
from torch import Tensor
@dataclass(frozen=True)
class PlannedLaunches:
    source_slots: Tensor
    angle: Tensor
    ships: Tensor
    target_slots: Tensor
    eta_turns: Tensor
    valid: Tensor
    fleet_ids: Tensor
@dataclass(frozen=True)
class LaunchEntries:
    """Multi-launch table for one planning step.
    Each ``[L]`` entry encodes a single launch:
        ``source_slots[b, l]`` -> ``target_slots[b, l]`` with ``ships[b, l]``
        ships at heading ``angle[b, l]`` (rad), ETA ``eta[b, l]`` turns.
    Multiple entries may share the same ``source_slots`` value to encode
    multi-launch fan-out from a single planet. The per-source sum of
    ``ships`` over ``valid`` entries must respect that source's ship budget;
    the engine debits sources sequentially in entry order, so callers should
    plan against running residuals rather than the original budget.
    Entry order also defines the launch dispatch order — fleet IDs assigned
    via :func:`infer_planned_launches_from_entries` increase in cumulative
    order over valid entries, matching the engine's ``cumsum`` rule for
    sparse launch payloads.
    """
    source_slots: Tensor  # [L] long
    target_slots: Tensor  # [L] long
    ships: Tensor  # [L] float
    angle: Tensor  # [L] float
    eta: Tensor  # [L] float
    valid: Tensor  # [L] bool
    @property
    def width(self) -> int:
        return int(self.source_slots.shape[0])
def concat_launch_entries(entries: Sequence[LaunchEntries]) -> LaunchEntries:
    """Concatenate launch-entry tables along the L axis.
    All inputs must share the same ``B`` and per-tensor dtype/device.
    """
    if not entries:
        raise ValueError("concat_launch_entries requires at least one entry table")
    if len(entries) == 1:
        return entries[0]
    return LaunchEntries(
        source_slots=torch.cat([e.source_slots for e in entries], dim=0),
        target_slots=torch.cat([e.target_slots for e in entries], dim=0),
        ships=torch.cat([e.ships for e in entries], dim=0),
        angle=torch.cat([e.angle for e in entries], dim=0),
        eta=torch.cat([e.eta for e in entries], dim=0),
        valid=torch.cat([e.valid for e in entries], dim=0),
    )
def disambiguate_duplicate_launches(
    entries: LaunchEntries,
    *,
    epsilon: float = 1.0e-5,
) -> LaunchEntries:
    """Perturb angle on duplicate launches so they\'re tracker-distinguishable.
    The engine's slot-order fleet-id assignment plus the agent's
    reconciliation by ``(owner, source, ships, angle)`` cannot disambiguate
    two pending entries that share the full tuple, even though the engine
    creates two distinct fleets. ``PlanetMovement._reconcile_pending_own_launches``
    hard-fails on such collisions ("multiple pending entries resolved to the
    same engine fleet id …").
    This helper finds entries that share ``(source, angle, ships)`` with an
    earlier valid entry in the same lane and adds ``k * epsilon`` to the
    angle of the k-th duplicate. ``epsilon = 1e-5`` rad is well above
    float32's ULP at angle magnitude ~1 (≈6e-8) and well below any
    behaviorally-meaningful aim error (5e-4 unit displacement at 50-unit
    fleet range — sub-planet-radius).
    Both the engine action (``_entries_to_sparse_payload``) and the stash
    (``infer_planned_launches_from_entries``) read ``entries.angle``, so
    applying the perturbation here keeps both branches consistent — the
    engine creates fleets with the perturbed angle, the obs reports the
    perturbed angle, and the stash matches.
    """
    src = entries.source_slots                                                 # [L]
    ang = entries.angle                                                         # [L]
    ships = entries.ships                                                       # [L]
    valid = entries.valid                                                       # [L]
    L = src.shape[0]
    if L < 2 or not bool(valid.any()):
        return entries
    device = src.device
    src_i = src.unsqueeze(1)                                                    # [L, 1]
    src_j = src.unsqueeze(0)                                                    # [1, L]
    ang_i = ang.unsqueeze(1)
    ang_j = ang.unsqueeze(0)
    ships_i = ships.unsqueeze(1)
    ships_j = ships.unsqueeze(0)
    valid_i = valid.unsqueeze(1)
    valid_j = valid.unsqueeze(0)
    j_indices = torch.arange(L, device=device).view(1, L)
    i_indices = torch.arange(L, device=device).view(L, 1)
    earlier = j_indices < i_indices                                             # [L, L]
    match = (
        valid_i & valid_j
        & (src_i == src_j)
        & (ang_i == ang_j)
        & (ships_i == ships_j)
        & earlier
    )                                                                           # [L, L]
    if not bool(match.any()):
        return entries
    dup_count = match.sum(dim=1).to(ang.dtype)                                  # [L]
    new_angle = ang + dup_count * float(epsilon)
    return LaunchEntries(
        source_slots=entries.source_slots,
        target_slots=entries.target_slots,
        ships=entries.ships,
        angle=new_angle,
        eta=entries.eta,
        valid=entries.valid,
    )
def ensure_planet_movement(
    *,
    obs_tensors: dict,
    expected_cfg: MovementConfig,
    cached_movement: PlanetMovement | None,
) -> PlanetMovement:
    """Reuse the cached movement (rolled forward) if its config matches, else
    rebuild from the observation. Returns the live movement cache."""
    if cached_movement is not None and cached_movement.config == expected_cfg:
        cached_movement.update(obs_tensors)
        return cached_movement
    return PlanetMovement.from_obs_tensors(obs_tensors, config=expected_cfg)
def _resolve_player_next_fleet_id(
    obs_tensors: dict,
    *,
    device: torch.device,
) -> Tensor:
    next_fleet_id = obs_tensors.get("player_next_fleet_id", obs_tensors.get("next_fleet_id"))
    if next_fleet_id is None:
        return torch.zeros((), dtype=torch.long, device=device)
    return next_fleet_id.to(device=device, dtype=torch.long)
def infer_planned_launches_from_entries(
    *,
    obs_tensors: dict,
    movement: PlanetMovement,
    entries: LaunchEntries,
    player_id: int,
) -> PlannedLaunches:
    """Resolve fleet IDs and target/ETA arrivals for a launch table.
    Fleet IDs increase in entry order over valid launches via
    ``cumsum(valid) - valid``. This matches the engine's sparse rule and
    cleanly handles multi-launch from the same source slot (each entry receives
    a distinct fleet ID). Target/ETA are recomputed via the swept-pair physics
    in :func:`_estimate_new_fleet_arrivals`. Result is shaped ``[L]``.
    """
    source_slots = entries.source_slots
    angle = entries.angle
    ships = entries.ships
    launch_valid = entries.valid
    L = source_slots.shape[0]
    device = source_slots.device
    P = max(int(movement.P), 1)
    next_fleet_id = _resolve_player_next_fleet_id(obs_tensors, device=device)
    # ``cumsum(valid) - valid`` mirrors the engine's launch_rank formula and is
    # independent of source ordering, so it supports multi-launch per source.
    launch_long = launch_valid.to(torch.long)
    launch_rank = launch_long.cumsum(0) - launch_long
    fleet_ids = next_fleet_id + launch_rank
    src_safe = source_slots.clamp(min=0, max=P - 1)
    launch_x, launch_y = movement.position_at_slots(src_safe, 0)
    source_r = movement.radii[src_safe]
    start_x = launch_x + torch.cos(angle) * (source_r + 0.1)
    start_y = launch_y + torch.sin(angle) * (source_r + 0.1)
    source_planet_ids = movement.planet_ids[src_safe]
    rows = torch.full((L, 7), -1.0, dtype=movement.dtype, device=device)
    rows[..., 0] = fleet_ids.to(dtype=movement.dtype)
    rows[..., 1] = float(player_id)
    rows[..., 2] = start_x.to(dtype=movement.dtype)
    rows[..., 3] = start_y.to(dtype=movement.dtype)
    rows[..., 4] = angle.to(dtype=movement.dtype)
    rows[..., 5] = source_planet_ids.to(dtype=movement.dtype)
    rows[..., 6] = ships.to(dtype=movement.dtype)
    rows[..., 0] = torch.where(
        launch_valid, rows[..., 0], torch.full_like(rows[..., 0], -1.0)
    )
    target_slots = torch.zeros(L, dtype=torch.long, device=device)
    eta_turns = torch.zeros(L, dtype=torch.float32, device=device)
    intent_valid = torch.zeros(L, dtype=torch.bool, device=device)
    fleet_slot = torch.where(launch_valid)[0]
    if int(fleet_slot.numel()) > 0:
        estimate = _estimate_new_fleet_arrivals(
            movement=movement,
            obs_fleets=rows,
            fleet_slot=fleet_slot,
        )
        valid_hit = estimate["has_hit"]
        if bool(valid_hit.any()):
            src = fleet_slot[valid_hit]
            target_slots[src] = estimate["target_slot"][valid_hit]
            eta_turns[src] = estimate["eta_index"][valid_hit].to(dtype=torch.float32) + 1.0
            intent_valid[src] = True
    return PlannedLaunches(
        source_slots=source_slots,
        angle=angle,
        ships=ships,
        target_slots=target_slots,
        eta_turns=eta_turns,
        valid=intent_valid,
        fleet_ids=fleet_ids,
    )
def apply_private_planned_launches(
    *,
    movement: PlanetMovement,
    launches: PlannedLaunches,
    owner_id: int,
    obs_tensors: dict,
) -> None:
    """Record an agent\'s just-decided launches into its movement cache.
    Seeds the arrival buckets with the source-derived prediction but does *not*
    seed the ``tracked_fleet_ids`` ledger directly: ``launches.fleet_ids`` come
    from the global ``next_fleet_id`` plus a cumsum, which collides with other
    slots' IDs because the engine processes player actions in slot order.
    Instead the launches are stashed and paired against the next observation's
    fleets (which carry the engine's authoritative IDs) via
    ``_reconcile_pending_own_launches``.
    ``obs_tensors`` is required (we snapshot ``next_fleet_id`` for reconciliation).
    """
    if not movement.track_fleets:
        return
    movement.record_fleet_arrivals(
        target_slots=launches.target_slots,
        owner_ids=int(owner_id),
        ships=launches.ships,
        eta=launches.eta_turns,
        valid=launches.valid,
    )
    nfid = obs_tensors.get("next_fleet_id")
    if nfid is None:
        raise ValueError("obs_tensors is missing \'next_fleet_id\'")
    movement.stash_pending_own_launches(
        owner_id=int(owner_id),
        source_slots=launches.source_slots,
        ships=launches.ships,
        angle=launches.angle,
        target_slots=launches.target_slots,
        eta=launches.eta_turns,
        valid=launches.valid,
        prev_next_fleet_id=nfid,
    )
# ==========================================================================
# orbit_lite.planner_core
# ==========================================================================
"""Flow-diff scored planner core: candidate scoring, shortlists, aim, selection.
Pure, tensor-only planning helpers for one game: the competitive net-ship-delta
scorer, target/source shortlists, capture-floor sizing, the strict-superset
reachability gate, the device-stable greedy selector, the hold-reserve cap
``safe_drain``, and the pressure-gradient regrouper.
"""
import torch
from torch import Tensor
def largest_initial_player_count(obs_tensors: dict) -> int:
    """Player count for the match: metadata if present, else distinct initial owners.
    """
    metadata_count = obs_tensors.get("player_count")
    if metadata_count is not None:
        count = (
            int(metadata_count.flatten()[0].item())
            if isinstance(metadata_count, Tensor)
            else int(metadata_count)
        )
        if count in (2, 4):
            return count
    initial = obs_tensors["initial_planets"]      # [P, 7]
    pid = initial[:, 0]
    owner = initial[:, 1]
    mask = (pid >= 0) & (owner >= 0)
    owners = owner[mask]
    n_max = 2
    if owners.numel() > 0:
        n_max = max(n_max, int(torch.unique(owners.long()).numel()))
    return n_max
# ---------------------------------------------------------------------------
# Scoring (P2): candidate launches -> competitive net-ship-delta
# ---------------------------------------------------------------------------
def make_launch_set(
    *,
    source_slots: Tensor,   # [C, L] long
    target_slots: Tensor,   # [C, L] long
    ships: Tensor,          # [C, L] float
    eta: Tensor,            # [C, L] float (steps to arrival, >= 1)
    valid: Tensor,          # [C, L] bool
    player_id: int,
) -> LaunchSet:
    """Build a candidate-axis ``LaunchSet`` owned by ``player_id``."""
    owner = torch.full_like(source_slots, int(player_id), dtype=torch.long)
    return LaunchSet(
        source_slots=source_slots.to(torch.long),
        target_slots=target_slots.to(torch.long),
        ships=ships,
        eta=eta,
        owner=owner,
        valid=valid.to(torch.bool),
    )
def competitive_score(diff: GarrisonFlowDiff, *, player_id: int) -> Tensor:
    """Competitive score: ``Δnet_me − Σ_opp Δnet_opp``.
    ``diff.net_ship_delta`` is ``[*prefix, A]`` (per-player change in net ships
    gained = produced − lost-to-combat); returns ``[*prefix]``. The opponent term
    is the equal-weight sum over rivals, so a launch is worth my net gain minus
    the opponents' net gain.
    """
    net = diff.net_ship_delta                       # [*prefix, A]
    me = net[..., int(player_id)]
    opp = net.sum(dim=-1) - me
    return me - opp
def score_candidates(
    status: PlanetGarrisonStatus,
    *,
    prod: Tensor,
    alive_by_step: Tensor,
    player_count: int,
    launches: LaunchSet,
    player_id: int,
) -> Tensor:
    """Competitive score per candidate. ``[C]`` (or scalar if no candidate axis).
    Uses the sparse exact flow projector.
    """
    diff = sparse_launch_flow_delta(
        status,
        prod=prod,
        alive_by_step=alive_by_step,
        player_count=int(player_count),
        launches=launches,
        player_id=int(player_id),
    )
    return competitive_score(diff, player_id=int(player_id))
# ---------------------------------------------------------------------------
# Candidate generation + greedy selection (P3: single-source, single-k, attack)
# ---------------------------------------------------------------------------
# Selection on CPU and CUDA must agree exactly: `torch.topk` / `torch.argmax`
# break ties differently across devices, and this planner ranks by integer ship
# counts / proximity that tie constantly — so device-stable selection is what
# keeps batch-CUDA play identical to CPU. We break ties by ascending slot index
# on both devices via a stable sort / lowest-index argmax.
def _stable_topk_indices(ranked: Tensor, k: int) -> Tensor:
    """Indices of the top-``k`` along the last dim, ties broken by ascending index
    identically on CPU and CUDA (stable descending sort)."""
    order = torch.argsort(ranked, dim=-1, descending=True, stable=True)
    return order[..., :max(1, int(k))]
def _stable_argmax(scores: Tensor) -> Tensor:
    """Lowest-index argmax along the last dim, device-deterministic on ties."""
    C = int(scores.shape[-1])
    is_max = scores == scores.max(dim=-1, keepdim=True).values
    idx = torch.arange(C, device=scores.device).expand_as(scores)
    return torch.where(is_max, idx, torch.full_like(idx, C)).argmin(dim=-1)
def _candidate_indices(values: Tensor, mask: Tensor, cap: int) -> tuple[Tensor, Tensor]:
    """Top-``cap`` slot indices of ``values`` under ``mask``. ``([K] long, [K] bool)``.
    Device-stable (ascending-index tie-break) — see note above.
    """
    p_count = values.shape[0]
    k = p_count if cap <= 0 else min(int(cap), p_count)
    neg_inf = torch.full_like(values, float("-inf"))
    ranked = torch.where(mask, values, neg_inf)
    top_idx = _stable_topk_indices(ranked, max(1, k))
    top_vals = ranked[top_idx]
    return top_idx, top_vals > float("-inf")
def is_comet_planet(obs_tensors: dict, P: int, device: torch.device) -> Tensor | None:
    """Per-slot mask of active comet planets, or ``None`` if absent."""
    comet_ids = obs_tensors.get("comet_planet_ids")
    planets = obs_tensors.get("planets")
    if comet_ids is None or planets is None:
        return None
    planet_ids = planets[..., 0].long()                       # [P]
    comet_ids = comet_ids.to(device=device)
    mask = torch.zeros(P, dtype=torch.bool, device=device)
    for c in range(int(comet_ids.shape[-1])):
        cid = comet_ids[c]
        mask = mask | ((planet_ids == cid) & (cid >= 0))
    return mask
def reinforcement_timing_factor(
    eta: Tensor,
    *,
    eta_free: float,
    eta_scale: float,
) -> Tensor:
    """Reaction-likelihood ramp ``ρ(eta) ∈ [0, 1]`` for reinforcement risk.
    ``ρ = clamp((eta − eta_free) / eta_scale, 0, 1)``. Below ``eta_free`` turns of
    flight the enemy has no time to react (ρ=0); over the next ``eta_scale`` turns
    reaction likelihood ramps linearly to 1. Pure arithmetic → CPU/CUDA agree.
    """
    scale = max(float(eta_scale), 1e-6)
    return ((eta - float(eta_free)) / scale).clamp(0.0, 1.0)
def capture_floor(
    garrison_status: PlanetGarrisonStatus,
    *,
    target_idx: Tensor,        # [T] long
    k_max: int,
    capture_overhead: float,
    player_id: int,
    reinforcement: Tensor | None = None,   # [T, K'>=K] float; added before ceil
) -> Tensor:
    """Owner-aware send floor per target at arrival turn ``k``. ``[T, K]``.
    - If I **own** the target at ``k`` (reinforcement), the floor is 1 — arriving
      ships add to my garrison, there is nothing to clear.
    - Otherwise (capture / retake), the floor is ``ceil(projected_defenders_at_k +
      overhead)``.
    - ``reinforcement`` (optional, ``[T, K' ≥ K]``) is added to the defender count
      before the ceil on capture cells (not on ``mine_at_k`` reinforcement cells) —
      the ETA-aware reactive-reinforcement margin. ``None`` ⇒ today's behaviour.
    Assumes ``k_max <= H``.
    """
    ships = garrison_status.ships
    owner = garrison_status.owner
    dtype = ships.dtype if ships.is_floating_point() else torch.float32
    T = target_idx.shape[0]
    H_axis = int(ships.shape[-1])
    P = int(ships.shape[0])
    K = max(0, min(int(k_max), H_axis - 1))
    if K == 0:
        return torch.empty(T, 0, dtype=dtype, device=ships.device)
    tgt = target_idx.clamp(min=0, max=max(P - 1, 0))
    gathered = ships[tgt].to(dtype=dtype)                       # [T, H+1]
    owner_g = owner[tgt]                                        # [T, H+1]
    k_idx = torch.arange(1, K + 1, device=ships.device).view(1, K).expand(T, K)
    defenders = gathered.gather(-1, k_idx)                      # [T, K]
    mine_at_k = owner_g.gather(-1, k_idx) == int(player_id)
    if reinforcement is not None:
        # Caller passes a margin with K' >= K (built from k_max=K_eta, while this
        # function's K = min(k_max, H-1) <= K_eta); slice down to our own K.
        assert reinforcement.shape[-1] >= K, (
            f"reinforcement last dim {reinforcement.shape[-1]} < capture_floor K={K}"
        )
        extra = reinforcement[..., :K].to(dtype=dtype, device=ships.device)
    else:
        extra = 0.0
    cap = (defenders + float(capture_overhead) + extra).clamp(min=1.0).ceil()
    return torch.where(mine_at_k, torch.ones_like(cap), cap)
def attack_target_mask(obs, obs_tensors: dict) -> Tensor:
    """Enemy ∪ neutral, alive, non-comet. ``[P]`` bool."""
    mask = (obs.is_enemy | obs.is_neutral) & obs.alive
    comet = is_comet_planet(obs_tensors, obs.P, obs.device)
    if comet is not None:
        mask = mask & ~comet
    return mask
def friendly_flip_targets(
    obs, garrison_status: PlanetGarrisonStatus, *, H: int, prod: Tensor,
) -> tuple[Tensor, Tensor]:
    """Own planets the do-nothing projection shows flipping within H.
    Returns ``(mask [P] bool, urgency [P] float)``. ``urgency`` ≈ projected
    ships lost if unaddressed = ``prod·(H − flip_turn) + garrison_now`` — same ship
    units as the ROI, used to fill the reserved defensive sub-quota.
    """
    P = obs.P
    device = obs.device
    pid = int(obs.player_id)
    if H <= 0:
        z = torch.zeros(P, device=device)
        return torch.zeros(P, dtype=torch.bool, device=device), z
    owner_h = garrison_status.owner[..., 1:]                     # [P, H]
    flips = obs.owned.unsqueeze(-1) & (owner_h != pid)           # currently mine, not mine at some k
    any_flip = flips.any(dim=-1)                                 # [P]
    # earliest flip turn (lowest-index True); _stable_argmax instead of raw argmax
    # so the tie among post-flip turns resolves identically on CPU and CUDA.
    flip_turn = _stable_argmax(flips.to(torch.int64)) + 1        # 1-based; valid where any_flip
    remaining = (float(H) - flip_turn.to(prod.dtype)).clamp(min=0.0)
    urgency = prod * remaining + obs.ships
    urgency = torch.where(any_flip, urgency, torch.full_like(urgency, float("-inf")))
    return any_flip, urgency
def ffa_target_bias(obs, *, config, player_count: int) -> Tensor:
    """4P-only strategic target bias in score units, one value per planet.
    Keeps the tactical flow scorer intact, but nudges target discovery/selection
    toward valuable neutrals and leader-owned planets in FFA.
    """
    P = obs.P
    device = obs.device
    dtype = obs.ships.dtype
    if int(player_count) < 4:
        return torch.zeros(P, dtype=dtype, device=device)
    weights_on = (
        float(config.ffa_neutral_prod_weight) != 0.0
        or float(config.ffa_enemy_prod_weight) != 0.0
        or float(config.ffa_leader_focus_bonus) != 0.0
        or float(config.ffa_weak_enemy_penalty) != 0.0
    )
    if not weights_on:
        return torch.zeros(P, dtype=dtype, device=device)
    A_eff = max(1, int(player_count))
    owner_raw = obs.owner_abs.to(device=device, dtype=torch.long)
    valid_owner = obs.alive & (owner_raw >= 0) & (owner_raw < A_eff)
    owner_safe = owner_raw.clamp(0, A_eff - 1)
    prod_by_owner = torch.zeros(A_eff, dtype=dtype, device=device)
    prod_by_owner.scatter_add_(0, owner_safe, torch.where(valid_owner, obs.prod.to(dtype), torch.zeros(P, dtype=dtype, device=device)))
    pid = int(obs.player_id)
    player_idx = torch.arange(A_eff, device=device)
    enemy_player = player_idx != pid
    enemy_prod = torch.where(enemy_player, prod_by_owner, torch.full_like(prod_by_owner, float("-inf")))
    leader_prod = enemy_prod.max()
    my_prod = prod_by_owner[pid] if 0 <= pid < A_eff else torch.zeros((), dtype=dtype, device=device)
    owner_prod = prod_by_owner[owner_safe]
    is_enemy = obs.is_enemy & valid_owner
    leader_target = is_enemy & (owner_prod >= leader_prod - 1e-6) & (
        leader_prod >= my_prod + float(config.ffa_leader_prod_gap)
    )
    weak_enemy = is_enemy & (owner_prod + float(config.ffa_leader_prod_gap) < leader_prod)
    bias = torch.zeros(P, dtype=dtype, device=device)
    bias = bias + obs.is_neutral.to(dtype) * obs.prod.to(dtype) * float(config.ffa_neutral_prod_weight)
    bias = bias + is_enemy.to(dtype) * obs.prod.to(dtype) * float(config.ffa_enemy_prod_weight)
    bias = bias + leader_target.to(dtype) * float(config.ffa_leader_focus_bonus)
    bias = bias - weak_enemy.to(dtype) * float(config.ffa_weak_enemy_penalty)
    return torch.where(obs.alive, bias, torch.zeros_like(bias))
def build_target_shortlist(
    obs, obs_tensors, garrison_status, cache, *, config, K_eta, H, prod, source_mask, player_count,
):
    """Single unified shortlist: ``max_offensive_targets`` enemy/neutral targets by
    proximity ∪ ``max_defensive_targets`` friendly-flip targets by urgency., The
    two caps are independent (shortlist width == offensive + defensive), so each can
    be swept on its own. Returns ``(target_idx, target_exists)``."""
    P = obs.P
    device = obs.device
    n_attack = max(1, min(int(config.max_offensive_targets), P))
    R = max(0, min(int(config.max_defensive_targets), P))
    attack_mask = attack_target_mask(obs, obs_tensors)
    proximity = min_distance_to_targets(cache, source_mask, attack_mask, max_k=K_eta)
    bias = ffa_target_bias(obs, config=config, player_count=int(player_count))
    attack_pref = torch.where(attack_mask, -proximity + bias, torch.full_like(proximity, float("-inf")))
    atk_idx, atk_exists = _candidate_indices(attack_pref, attack_mask, n_attack)
    if R > 0:
        flip_mask, urgency = friendly_flip_targets(obs, garrison_status, H=H, prod=prod)
        def_idx, def_exists = _candidate_indices(urgency, flip_mask, R)
        target_idx = torch.cat([atk_idx, def_idx], dim=0)
        target_exists = torch.cat([atk_exists, def_exists], dim=0)
    else:
        target_idx, target_exists = atk_idx, atk_exists
    return target_idx, target_exists
def reachable_mask(
    movement: PlanetMovement,
    *,
    source_idx: Tensor,      # [S] long
    target_idx: Tensor,      # [T] long
    fleet_sizes: Tensor,     # [S, T, G] float
    eta_cap: Tensor,         # [T] float (per-target reach cap)
    eps: float = 1e-4,
) -> Tensor:
    """Strict-superset reachability gate for the body screen, ``[S, T, G]`` bool.
    A cell is reachable iff some step interval ``k in [1, eta_cap[b,t]]`` admits the
    straight-line shot: ``(d_k - gap) <= fleet_speed(size) * k * (1 + eps)`` where
    ``d_k`` is the distance from the source centre @ turn 0 to the target's **swept
    segment** ``[tgt@(k-1), tgt@k]`` and ``gap = src_r + tgt_r + offsets``.
    Using the swept segment (not the point ``tgt@k``) and the surface gap makes this
    a provable *necessary condition* for ``intercept_angle`` viability: a viable shot
    contacts the target at some continuous ``t_c <= eta_cap`` with
    ``dist(src@0, tgt@t_c) - gap <= speed * t_c <= speed * ceil(t_c)``, and the
    segment distance over the interval containing ``t_c`` is ``<= dist(src@0, tgt@t_c)``.
    Hence ``viable => reachable`` (the ``eps`` absorbs fp32 boundary noise) — the gate
    never false-prunes a launch the agent would otherwise aim. ``intercept_angle``
    re-validates every survivor, so the surplus kept beyond true viability is harmless.
    """
    S, T, G = fleet_sizes.shape
    P = int(movement.P)
    dt = movement.dtype
    K = max(1, min(int(movement.movement_horizon), int(torch.ceil(eta_cap.max()).item())))
    src = source_idx.clamp(0, P - 1)
    tgt = target_idx.clamp(0, P - 1)
    # Source centre @ turn 0; target positions @ turns 0..K (segment endpoints).
    sx = movement.x[0][src].view(S, 1, 1)                                   # [S,1,1]
    sy = movement.y[0][src].view(S, 1, 1)
    tx = movement.x[: K + 1].gather(1, tgt.view(1, T).expand(K + 1, T))     # [K+1,T]
    ty = movement.y[: K + 1].gather(1, tgt.view(1, T).expand(K + 1, T))
    ax = tx[:K, :].view(1, K, T); ay = ty[:K, :].view(1, K, T)             # tgt@(k-1)
    bx = tx[1:, :].view(1, K, T); by = ty[1:, :].view(1, K, T)             # tgt@k
    # Point-to-segment distance from (sx,sy) to segment [(ax,ay),(bx,by)] → [S,K,T].
    abx = bx - ax; aby = by - ay
    apx = sx - ax; apy = sy - ay
    denom = (abx * abx + aby * aby).clamp(min=1e-12)
    u = ((apx * abx + apy * aby) / denom).clamp(0.0, 1.0)
    cx = ax + u * abx; cy = ay + u * aby
    seg_dist = torch.sqrt(((sx - cx) ** 2 + (sy - cy) ** 2).clamp(min=0.0))  # [S,K,T]
    src_r = movement.radii[src].view(S, 1, 1)
    tgt_r = movement.radii[tgt].view(1, 1, T)
    gap = src_r + tgt_r + (LAUNCH_SURFACE_OFFSET + TARGET_HIT_SURFACE_OFFSET)
    surf = (seg_dist - gap).clamp(min=0.0)                                   # [S,K,T]
    kv = torch.arange(1, K + 1, device=movement.device, dtype=dt).view(1, K, 1)
    ratio = surf / kv
    within = kv <= eta_cap.view(1, 1, T)                                    # [1,K,T]
    ratio = torch.where(within, ratio, torch.full_like(ratio, float("inf")))
    min_ratio = ratio.amin(dim=1)                                          # [S,T]
    speed = fleet_speed(fleet_sizes.clamp(min=1.0))                          # [S,T,G]
    reachable = min_ratio.unsqueeze(-1) <= speed * (1.0 + float(eps))        # [S,T,G]
    distinct = (src.view(S, 1) != tgt.view(1, T)).unsqueeze(-1)             # [S,T,1]
    return reachable & distinct
def _greedy_select(
    *, P, W, device, dtype, score, cand_src, cand_send, cand_angle, cand_eta,
    cand_active, cand_tgt_slot, cand_tgt_short, cand_is_def, source_budget,
    target_exists, roi_threshold, ship_spend_penalty=0.0, defense_spend_discount=0.25,
    lookahead_weight=0.0,
) -> LaunchEntries:
    """Masking-only greedy over [C, L] candidates: pick the best wave each iter,
    one per target, source-budget aware across all L contributors. Enforces the
    role mutex: a reinforced planet can\'t also be a source, and vice-versa."""
    C, L = int(cand_src.shape[0]), int(cand_src.shape[1])
    target_taken = ~target_exists.clone()                                        # [T]
    defended = torch.zeros(P, dtype=torch.bool, device=device)                   # reinforced this turn
    used_src = torch.zeros(P, dtype=torch.bool, device=device)                   # contributed this turn
    w_src = torch.zeros(W, L, dtype=torch.long, device=device)
    w_send = torch.zeros(W, L, dtype=dtype, device=device)
    w_angle = torch.zeros(W, L, dtype=dtype, device=device)
    w_eta = torch.ones(W, L, dtype=dtype, device=device)
    w_tgt = torch.zeros(W, L, dtype=torch.long, device=device)
    w_active = torch.zeros(W, L, dtype=torch.bool, device=device)
    for w in range(W):
        taken_cand = target_taken[cand_tgt_short]                               # [C]
        budget_at = source_budget[cand_src]                                     # [C, L]
        can_fund = ((cand_send <= budget_at) | ~cand_active).all(dim=-1)        # [C]
        # role mutex: target not already drained as a source; no contributor is a
        # planet we're reinforcing this turn.
        tgt_used_as_src = used_src[cand_tgt_slot]                               # [C]
        contrib_defended = (defended[cand_src] & cand_active).any(dim=-1)       # [C]
        mask = torch.isfinite(score) & ~taken_cand & can_fund & ~tgt_used_as_src & ~contrib_defended
        total_send = torch.where(cand_active, cand_send, torch.zeros_like(cand_send)).sum(dim=-1)
        spend_discount = torch.where(
            cand_is_def,
            torch.full_like(score, float(defense_spend_discount)),
            torch.ones_like(score),
        )
        effective_score = score - float(ship_spend_penalty) * total_send * spend_discount
        masked_current = torch.where(mask, effective_score, torch.full_like(score, float("-inf")))
        if float(lookahead_weight) > 0.0 and C > 1:
            cand_debit = torch.zeros(C, P, dtype=dtype, device=device)
            cand_debit.scatter_add_(
                1,
                cand_src,
                torch.where(cand_active, cand_send, torch.zeros_like(cand_send)),
            )
            budget_after_i = (source_budget.view(1, P) - cand_debit).clamp(min=0.0)
            budget_for_j_after_i = budget_after_i[:, cand_src]                  # [i, j, l]
            j_can_fund_after_i = (
                (cand_send.view(1, C, L) <= budget_for_j_after_i)
                | ~cand_active.view(1, C, L)
            ).all(dim=-1)                                                       # [i, j]
            i_used_src = cand_debit > 0.0                                       # [i, p]
            j_tgt_used_by_i = i_used_src[:, cand_tgt_slot]                      # [i, j]
            j_uses_i_defended_tgt = (
                (
                    cand_src.view(1, C, L) == cand_tgt_slot.view(C, 1, 1)
                )
                & cand_active.view(1, C, L)
            ).any(dim=-1) & cand_is_def.view(C, 1)                              # [i, j]
            distinct_target = cand_tgt_short.view(1, C) != cand_tgt_short.view(C, 1)
            next_mask = (
                mask.view(1, C)
                & distinct_target
                & j_can_fund_after_i
                & ~j_tgt_used_by_i
                & ~j_uses_i_defended_tgt
            )
            next_score = torch.where(
                next_mask,
                effective_score.view(1, C),
                torch.full((C, C), float("-inf"), dtype=dtype, device=device),
            ).max(dim=-1).values
            next_bonus = (next_score - float(roi_threshold)).clamp(min=0.0)
            masked_rank = torch.where(
                mask,
                effective_score + float(lookahead_weight) * next_bonus,
                torch.full_like(score, float("-inf")),
            )
        else:
            masked_rank = masked_current
        best_c = _stable_argmax(masked_rank)                                    # scalar, device-stable
        best_score = masked_current[best_c]
        fired = bool(torch.isfinite(best_score) & (best_score > roi_threshold))
        if not fired:
            break
        sel_src = cand_src[best_c]                   # [L]
        sel_send = cand_send[best_c]
        sel_active = cand_active[best_c]
        w_src[w] = sel_src
        w_send[w] = torch.where(sel_active, sel_send, torch.zeros_like(sel_send))
        w_angle[w] = cand_angle[best_c]
        w_eta[w] = cand_eta[best_c]
        w_tgt[w] = cand_tgt_slot[best_c]
        w_active[w] = sel_active
        # debit all contributors' sends from their source budgets.
        debit = torch.zeros_like(source_budget)
        debit.scatter_add_(0, sel_src, torch.where(sel_active, sel_send, torch.zeros_like(sel_send)))
        source_budget = (source_budget - debit).clamp(min=0.0)
        # mark target taken (one wave per target).
        target_taken[cand_tgt_short[best_c]] = True
        # role mutex bookkeeping: mark contributors used; mark reinforced targets
        # defended. Sum active marks per planet (order-independent) and OR them in.
        src_mark = torch.zeros(P, dtype=torch.long, device=device)
        src_mark.scatter_add_(0, sel_src, sel_active.to(torch.long))
        used_src = used_src | (src_mark > 0)
        sel_tgt = cand_tgt_slot[best_c]
        sel_is_def = bool(cand_is_def[best_c])
        defended[sel_tgt] = defended[sel_tgt] | sel_is_def
    # Flatten waves x contributors into a LaunchEntries table.
    WL = W * L
    entries = LaunchEntries(
        source_slots=w_src.reshape(WL),
        target_slots=w_tgt.reshape(WL),
        ships=torch.where(w_active, w_send, torch.zeros_like(w_send)).reshape(WL),
        angle=torch.where(w_active, w_angle, torch.zeros_like(w_angle)).reshape(WL),
        eta=torch.where(w_active, w_eta, torch.ones_like(w_eta)).reshape(WL),
        valid=w_active.reshape(WL),
    )
    return entries, source_budget   # source_budget = leftover ships per planet
def _plan_regroup(
    *, movement, obs, obs_tensors, garrison_status, leftover, original_ships,
    pressure, config, H,
) -> LaunchEntries:
    """Pressure-gradient marshalling of leftover ships.
    Moves ships from low-pressure planets toward nearby higher-pressure owned
    planets, capped by ``safe_drain`` (minus what attacks already drew), only when
    the destination is materially more stressed, reachable within
    ``max_regroup_time``, and **still owned at the fleet's arrival turn**.
    """
    P = obs.P
    device = obs.device
    dtype = original_ships.dtype
    pid = int(obs.player_id)
    min_send = float(config.min_ships_to_launch)
    src_mask = obs.owned & obs.alive & (leftover >= min_send)
    if not bool(src_mask.any()):
        return _empty_entries(device, dtype)
    S_cap = max(1, min(int(config.max_regroup_sources_per_lane), P))
    src_idx, src_exists = _candidate_indices(leftover, src_mask, S_cap)          # rank by leftover
    S = int(src_idx.shape[0])
    leftover_s = leftover[src_idx.clamp(0, P - 1)]
    orig_s = original_ships[src_idx.clamp(0, P - 1)]
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain_s = safe_drain(
        garrison_status, source_idx=src_idx, source_ships=orig_s,
        H_eff=H_eff, player_id=pid,
    )
    committed_s = (orig_s - leftover_s).clamp(min=0.0)
    regroup_cap = torch.minimum(leftover_s, (drain_s - committed_s).clamp(min=0.0)).floor()
    can_send = src_exists & (regroup_cap >= min_send)
    if not bool(can_send.any()):
        return _empty_entries(device, dtype)
    # Destinations are owned, alive, non-comet planets (do-nothing projection).
    dst_mask = obs.owned & obs.alive
    comet = is_comet_planet(obs_tensors, P, device)
    if comet is not None:
        dst_mask = dst_mask & ~comet
    T_cap = max(1, min(int(config.max_regroup_targets_per_source), P))
    dst_idx, dst_exists = _candidate_indices(pressure, dst_mask, T_cap)          # rank by pressure
    T = int(dst_idx.shape[0])
    # Fixed-size regroup aim via the continuous-intercept aimer (sub-turn lead + a
    # swept first-contact body screen on an AABB-culled shortlist).
    # Strict-superset reachability precheck defers the body screen to destinations a
    # source can reach within max_regroup_time (bit-identical to the ungated path).
    regroup_active = reachable_mask(
        movement, source_idx=src_idx, target_idx=dst_idx,
        fleet_sizes=regroup_cap.view(S, 1, 1).expand(S, T, 1),
        eta_cap=torch.full((T,), float(config.max_regroup_time), device=device),
    ).squeeze(-1)                                                                # [S, T]
    aim = intercept_angle(
        movement,
        src_idx.unsqueeze(1),                                                    # [S, 1]
        dst_idx.unsqueeze(0),                                                     # [1, T]
        regroup_cap.unsqueeze(1),                                                 # [S, 1]
        active=regroup_active,
    )
    angle = aim["angle"]                                                         # [S, T]
    eta = aim["eta"]
    viable = aim["viable"]
    src_pres = pressure[src_idx.clamp(0, P - 1)].view(S, 1)
    dst_pres = pressure[dst_idx.clamp(0, P - 1)].view(1, T)
    gap = dst_pres - src_pres                                                    # [S, T]
    # arrival-turn ownership check: dst must still be mine at k = ceil(eta).
    owner = garrison_status.owner                                               # [P, H+1]
    H_axis = int(owner.shape[-1])
    dst_owner = owner[dst_idx.clamp(0, P - 1)]                                  # [T, H+1]
    k = torch.ceil(eta).clamp(min=0, max=H_axis - 1).to(torch.long)             # [S, T]
    owner_at_k = dst_owner.unsqueeze(0).expand(S, T, H_axis).gather(-1, k.unsqueeze(-1)).squeeze(-1)
    still_mine = owner_at_k == pid
    src_neq_dst = src_idx.view(S, 1) != dst_idx.view(1, T)
    valid = (
        viable & still_mine & src_neq_dst
        & (gap > float(config.regroup_pressure_delta_min))
        & (eta <= float(config.max_regroup_time))
        & can_send.view(S, 1) & dst_exists.view(1, T)
    )
    sc = torch.where(
        valid,
        gap - float(config.regroup_time_penalty_weight) * eta,
        torch.full_like(gap, float("-inf")),
    )
    best_t = _stable_argmax(sc)                                                  # [S] device-stable
    best_score = sc.gather(-1, best_t.unsqueeze(-1)).squeeze(-1)                 # [S]
    best_valid = torch.isfinite(best_score)
    s_ar = torch.arange(S, device=device)
    best_dst = dst_idx[best_t]                                                   # [S]
    best_angle = angle[s_ar, best_t]
    best_eta = eta[s_ar, best_t]
    return LaunchEntries(
        source_slots=src_idx,
        target_slots=best_dst,
        ships=torch.where(best_valid, regroup_cap, torch.zeros_like(regroup_cap)),
        angle=torch.where(best_valid, best_angle, torch.zeros_like(best_angle)),
        eta=torch.where(best_valid, best_eta, torch.ones_like(best_eta)),
        valid=best_valid,
    )
def _empty_entries(device: torch.device, dtype: torch.dtype) -> LaunchEntries:
    z = torch.zeros(0, dtype=dtype, device=device)
    zl = torch.zeros(0, dtype=torch.long, device=device)
    return LaunchEntries(
        source_slots=zl, target_slots=zl, ships=z, angle=z, eta=z,
        valid=torch.zeros(0, dtype=torch.bool, device=device),
    )
def entries_to_sparse_payload(entries: LaunchEntries, *, planet_ids: Tensor) -> dict[str, Tensor]:
    """Convert a LaunchEntries table to the sparse action-row payload."""
    L = entries.source_slots.shape[0]
    device = entries.source_slots.device
    P = int(planet_ids.shape[0])
    valid_long = entries.valid.to(torch.int64)
    counts = valid_long.sum().to(torch.int32)
    max_count = int(counts.item())
    out_from = torch.full((max_count,), -1, dtype=torch.int32, device=device)
    out_angle = torch.zeros((max_count,), dtype=torch.float32, device=device)
    out_ships = torch.zeros((max_count,), dtype=torch.float32, device=device)
    if max_count == 0:
        return {"from_planet_id": out_from, "angle": out_angle, "num_ships": out_ships, "counts": counts}
    safe_src = entries.source_slots.clamp(min=0, max=max(P - 1, 0))
    from_pid_full = planet_ids[safe_src].to(torch.int32)
    launch_rank = valid_long.cumsum(0) - valid_long
    l_idx = torch.where(entries.valid)[0]
    pos = launch_rank[l_idx]
    out_from[pos] = from_pid_full[l_idx]
    out_angle[pos] = entries.angle[l_idx].to(torch.float32)
    out_ships[pos] = entries.ships[l_idx].to(torch.float32)
    return {"from_planet_id": out_from, "angle": out_angle, "num_ships": out_ships, "counts": counts}
def empty_action_row(device: torch.device) -> dict[str, Tensor]:
    """Sparse launch payload with zero launches."""
    return {
        "from_planet_id": torch.full((0,), -1, dtype=torch.int32, device=device),
        "angle": torch.zeros((0,), dtype=torch.float32, device=device),
        "num_ships": torch.zeros((0,), dtype=torch.float32, device=device),
        "counts": torch.zeros((), dtype=torch.int32, device=device),
    }
def safe_drain(
    garrison_status: PlanetGarrisonStatus,
    *,
    source_idx: Tensor,            # [S] long — planet slots to evaluate
    source_ships: Tensor,          # [S] float — current garrison at those slots
    H_eff: Tensor,                 # scalar float — horizon to protect the source over
    player_id: int = 0,
) -> Tensor:
    """Max ships a source can shed while staying held over ``H_eff``. ``[S]``.
    Closed form, no scoring. For every source slot, over the turns ``t = 1..H``
    where the do-nothing projection still has us holding the planet (``owner == me``,
    ``ships > 0``) within ``H_eff``, the largest amount we can remove now while the
    projected garrison stays non-negative on every such turn is
    ``min_t(ships_traj[t])`` — leaving the planet at 0 ships on the worst held turn
    is allowed. Capped by ``source_ships`` (can't send more than we hold now):
        safe_drain = clamp(min(min_t held(ships_traj), source_ships), 0)
    A *doomed* source (no turn is held within ``H_eff``) has nothing to protect:
    ``min_slack`` is ``+inf`` and the cap collapses to ``source_ships`` naturally.
    """
    S = source_idx.shape[0]
    ships_cache = garrison_status.ships
    dtype = ships_cache.dtype if ships_cache.is_floating_point() else torch.float32
    device = ships_cache.device
    H_axis = int(ships_cache.shape[-1])
    H = max(H_axis - 1, 0)
    P = int(ships_cache.shape[0])
    if H == 0:
        return torch.zeros(S, dtype=dtype, device=device)
    src_idx_safe = source_idx.clamp(min=0, max=max(P - 1, 0))
    src_ships_traj = ships_cache[src_idx_safe][..., 1:].to(dtype=dtype)          # [S, H]
    src_owner_traj = garrison_status.owner[src_idx_safe][..., 1:]                 # [S, H]
    me_owned = src_owner_traj == int(player_id)
    turn_grid = torch.arange(1, H + 1, device=device, dtype=dtype).view(1, H)
    within_horizon = turn_grid <= H_eff                                          # H_eff scalar
    held = me_owned & within_horizon & (src_ships_traj > 0.0)
    inf_fill = torch.full_like(src_ships_traj, float("inf"))
    cap_traj = torch.where(held, src_ships_traj, inf_fill)
    min_slack = cap_traj.min(dim=-1).values                                       # [S]
    return torch.minimum(min_slack, source_ships.to(dtype)).clamp(min=0.0)
# ==========================================================================
# orbit_lite.adapter
# ==========================================================================
"""Observation/action adapter between the move-list format and tensors.
Converts an observation dict (``{"planets": [...], "fleets": [...], ...}``) into
the named tensor observation the planner consumes, and converts the planner's
sparse launch payload
(``{"from_planet_id": [L], "angle": [L], "num_ships": [L], "counts": scalar}``)
back into a move list (``[[from_planet_id, angle, ships], ...]``).
"""
from typing import Any
import torch
def _infer_player_count_from_obs(planets: list[Any], fleets: list[Any], player_id: int) -> int:
    owners: list[int] = [int(player_id)]
    for row in planets:
        if len(row) >= 2 and int(row[0]) >= 0 and int(row[1]) >= 0:
            owners.append(int(row[1]))
    for row in fleets:
        if len(row) >= 2 and int(row[0]) >= 0 and int(row[1]) >= 0:
            owners.append(int(row[1]))
    return 4 if max(owners, default=0) >= 2 else 2
def _dict_obs_to_tensor(
    obs: dict[str, Any],
    player_id: int,
    P: int = P_MAX,
    F: int = F_MAX,
    device: Any = "cpu",
) -> dict[str, Any]:
    """Convert an observation dict to a single-game tensor observation.
    Input format::
        obs["planets"] = [[planet_id, owner, x, y, radius, ships, production], ...]
        obs["fleets"]  = [[fleet_id, owner, x, y, angle, from_id, ships], ...]
    Returns a tensor observation dict::
        "planets" : [P, 7]            "fleets" : [F, 7]
        "initial_planets" : [P, 7]    "comet_planet_ids" : [G*C]
        "comets" : nested padded tensors
        "player" / "angular_velocity" / "next_fleet_id" / "step" /
        "episode_steps" / "remainingOverageTime" : scalars
    """
    dev = torch.device(device)
    planets_raw = obs.get("planets", [])
    initial_planets_raw = obs.get("initial_planets", planets_raw)
    fleets_raw = obs.get("fleets", [])
    comets_raw = obs.get("comets", [])
    comet_planet_ids_raw = obs.get("comet_planet_ids", [])
    step = int(obs.get("step", 0))
    angvel = float(obs.get("angular_velocity", 0.03))
    max_steps = int(obs.get("episode_steps", DEFAULT_EPISODE_STEPS))
    remaining_overtime = float(obs.get("remainingOverageTime", 2.0))
    next_fleet_id = int(obs.get("next_fleet_id", 0))
    planet_t = torch.zeros(P, 7, dtype=torch.float32, device=dev)
    planet_t[..., 0] = -1.0
    for i, p in enumerate(planets_raw[:P]):
        pid, owner, x, y, r, ships, prod = p[:7]
        planet_t[i, 0] = float(pid)
        planet_t[i, 1] = float(owner)
        planet_t[i, 2] = float(x)
        planet_t[i, 3] = float(y)
        planet_t[i, 4] = float(r)
        planet_t[i, 5] = float(ships)
        planet_t[i, 6] = float(prod)
    initial_planet_t = torch.zeros(P, 7, dtype=torch.float32, device=dev)
    initial_planet_t[..., 0] = -1.0
    for i, p in enumerate(initial_planets_raw[:P]):
        pid, owner, x, y, r, ships, prod = p[:7]
        initial_planet_t[i, 0] = float(pid)
        initial_planet_t[i, 1] = float(owner)
        initial_planet_t[i, 2] = float(x)
        initial_planet_t[i, 3] = float(y)
        initial_planet_t[i, 4] = float(r)
        initial_planet_t[i, 5] = float(ships)
        initial_planet_t[i, 6] = float(prod)
    fleet_t = torch.zeros(F, 7, dtype=torch.float32, device=dev)
    fleet_t[..., 0] = -1.0
    fleet_t[..., 5] = -1.0
    for i, f in enumerate(fleets_raw[:F]):
        fid, owner, x, y, angle, from_id, ships = f[:7]
        fleet_t[i, 0] = float(fid)
        fleet_t[i, 1] = float(owner)
        fleet_t[i, 2] = float(x)
        fleet_t[i, 3] = float(y)
        fleet_t[i, 4] = float(angle)
        fleet_t[i, 5] = float(from_id)
        fleet_t[i, 6] = float(ships)
    comet_ids = torch.full((COMET_EVENTS, COMETS_PER_EVENT), -1, dtype=torch.int32, device=dev)
    comet_paths = torch.full(
        (COMET_EVENTS, COMETS_PER_EVENT, COMET_PATH_MAX, 2),
        float("nan"),
        dtype=torch.float32,
        device=dev,
    )
    comet_path_index = torch.full((COMET_EVENTS,), -1, dtype=torch.int32, device=dev)
    for group_idx, group in enumerate(comets_raw[:COMET_EVENTS]):
        comet_path_index[group_idx] = int(group.get("path_index", -1))
        group_ids = group.get("planet_ids", [])
        group_paths = group.get("paths", [])
        for comet_idx, pid in enumerate(group_ids[:COMETS_PER_EVENT]):
            comet_ids[group_idx, comet_idx] = int(pid)
        for comet_idx, path in enumerate(group_paths[:COMETS_PER_EVENT]):
            for point_idx, point in enumerate(path[:COMET_PATH_MAX]):
                comet_paths[group_idx, comet_idx, point_idx, 0] = float(point[0])
                comet_paths[group_idx, comet_idx, point_idx, 1] = float(point[1])
    comet_planet_ids = torch.full(
        (COMET_EVENTS * COMETS_PER_EVENT,),
        -1,
        dtype=torch.int32,
        device=dev,
    )
    for idx, pid in enumerate(comet_planet_ids_raw[: COMET_EVENTS * COMETS_PER_EVENT]):
        comet_planet_ids[idx] = int(pid)
    return {
        "planets": planet_t,
        "fleets": fleet_t,
        "player": torch.tensor(player_id, dtype=torch.int32, device=dev),
        "player_count": torch.tensor(_infer_player_count_from_obs(planets_raw, fleets_raw, player_id), dtype=torch.int32, device=dev),
        "angular_velocity": torch.tensor(angvel, dtype=torch.float32, device=dev),
        "initial_planets": initial_planet_t,
        "next_fleet_id": torch.tensor(next_fleet_id, dtype=torch.int32, device=dev),
        "comets": {
            "planet_ids": comet_ids,
            "paths": comet_paths,
            "path_index": comet_path_index,
        },
        "comet_planet_ids": comet_planet_ids,
        "step": torch.tensor(step, dtype=torch.int32, device=dev),
        "episode_steps": torch.tensor(max_steps, dtype=torch.int32, device=dev),
        "remainingOverageTime": torch.tensor(remaining_overtime, dtype=torch.float32, device=dev),
    }
def _sparse_actions_to_list(
    action_payload: dict[str, Any],
    obs: dict[str, Any],
    player_id: int,
) -> list[list[Any]]:
    # The payload is produced by ``entries_to_sparse_payload`` and is already a
    # well-formed sparse row: ``from_planet_id``/``angle``/``num_ships`` are rank-1
    # tensors and ``counts`` is a scalar count of active launches.
    from_pid_t = action_payload["from_planet_id"]
    angle_t = action_payload["angle"]
    num_ships_t = action_payload["num_ships"]
    counts = int(action_payload["counts"].item())
    planets_by_id = {int(p[0]): p for p in obs.get("planets", []) if len(p) >= 7}
    moves: list[list[Any]] = []
    for launch_idx in range(counts):
        from_pid = int(from_pid_t[launch_idx].item())
        ships = float(num_ships_t[launch_idx].item())
        angle = float(angle_t[launch_idx].item())
        if ships < 1.0:
            continue
        source = planets_by_id.get(from_pid)
        if source is None:
            continue
        owner = int(source[1])
        available = float(source[5])
        if owner != int(player_id):
            continue
        if ships != float(round(ships)) or ships > available:
            raise ValueError(
                "Invalid launch ship count in sparse action payload at "
                f"from_planet_id={from_pid}: requested={ships}, available={available}. "
                "Counts must be finite, integer-valued, >= 0, and <= available planet ships."
            )
        moves.append([from_pid, angle, int(ships)])
    return moves
def single_obs_to_tensor(
    obs: dict[str, Any],
    *,
    player_id: int,
    P: int = P_MAX,
    F: int = F_MAX,
    device: Any = "cpu",
) -> dict[str, Any]:
    """Public wrapper: convert one observation dict to a tensor observation."""
    return _dict_obs_to_tensor(obs, player_id=player_id, P=P, F=F, device=device)
def sparse_action_row_to_moves(
    action_payload: dict[str, Any],
    obs: dict[str, Any],
    *,
    player_id: int,
) -> list[list[Any]]:
    """Decode a sparse launch payload into a move list.
    The payload may contain multiple entries from the same source planet — each
    valid entry produces a ``[from_planet_id, angle, ships]`` move in iteration
    order, mirroring how the engine processes sparse rows.
    """
    return _sparse_actions_to_list(action_payload, obs, player_id=int(player_id))
# ==========================================================================
# notebook main.py (ProducerLite runtime + agent)
# ==========================================================================
import dataclasses
import os
import sys
from dataclasses import dataclass
# Make the sibling ``orbit_lite`` package importable wherever this file runs:
# loaded in place, dropped at a submission-archive root, or exec'd by
# kaggle_environments with no ``__file__`` (fall back to the working dir).
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import torch
from torch import Tensor
@dataclass(frozen=True)
class ProducerLiteConfig:
    """Behaviour knobs.  """
    # the projection window, the movement build length, AND the target ETA cap 
    horizon: int = 18
    # --- shortlists ------------------------------------------------------
    max_sources_per_lane: int = 12
    max_offensive_targets: int = 12         # enemy/neutral proximity targets
    max_defensive_targets: int = 4          
    # --- scoring / greedy ------------------------------------------------
    max_waves_per_turn: int = 6
    roi_threshold: float = 1.5              # fire if score > this
    min_ships_to_launch: float = 4.0
    enable_single_source_size_variants: bool = False
    single_source_size_fracs: tuple[float, ...] = (1.0,)
    greedy_ship_spend_penalty: float = 0.0
    greedy_defense_spend_discount: float = 0.25
    enable_greedy_lookahead: bool = False
    greedy_lookahead_weight: float = 0.0
    # --- regroup  ------------------------------
    enable_regroup: bool = True
    max_regroup_time: float = 7.0
    regroup_pressure_delta_min: float = 0.25
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 7
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 1e-3
    # --- potential-attack RISK_SCORE (added to the regroup gradient) -----
    enable_potential_risk: bool = False     # defense feature off by default (no measured gain)
    risk_blend_weight: float = 1.0          # weight of RISK added to cheap_enemy_pressure
    risk_enemy_prod_weight: float = 2.0     # enemy strength = ships + w * production
    risk_self_prod_weight: float = 2.0      # own-planet value  = 1 + w * production
    risk_support_weight: float = 0.5        # friendly-neighbour discount strength
    # --- 4P FFA strategic target shaping --------------------------------
    ffa_neutral_prod_weight: float = 0.0
    ffa_enemy_prod_weight: float = 0.0
    ffa_leader_focus_bonus: float = 0.0
    ffa_leader_prod_gap: float = 0.0
    ffa_weak_enemy_penalty: float = 0.0
    # --- focus fire (coordinated multi-source attack) -------------------
    enable_focus_fire: bool = True          # pool same-step sources to take strong targets
    max_strike_sources: int = 4             # max planets combined per strike (contributor cap L)
    enable_ppo_rerank: bool = False
    ppo_bonus_weight: float = 1.0
def _movement_config(config: ProducerLiteConfig, *, player_count: int) -> MovementConfig:
    """MovementConfig: fleet tracking on, horizon = config.horizon."""
    return MovementConfig(
        movement_horizon=int(config.horizon),
        drift_epsilon=1e-3,
        track_fleets=True,
        player_count=int(player_count),
        max_tracked_fleets=128,
    )
def cheap_enemy_pressure(obs, cache, *, horizon: float, player_id: int) -> Tensor:
    """Cheap reachable-enemy-mass proxy per planet — ``[P]``.
    Consumed only as the **regroup gradient** (rank owned planets by how stressed
    they are, move ships up the gradient). For each planet ``t``, sums a
    distance-decayed share of every enemy source's **current** garrison that could
    straight-line reach ``t`` within ``horizon`` turns, using the step-0 centre
    distance ``cross_dist[0]``. The decay ``(1 - d/(speed·H))₊`` weights nearer
    enemies more, giving a graded frontline signal in ship-mass units.
    Approximations: ignores target orbital drift over the horizon, production
    accrued in flight, the per-owner split, and in-flight enemy fleets. Pure
    arithmetic on cached tensors
    """
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    d0 = cache.cross_dist[0].to(dtype)                                   # [src, tgt] current centre dist
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))                          # [P]
    reach_dist = (speeds.view(P, 1) * float(horizon)).clamp(min=1e-6)    # [src, 1]
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))  # [P]
    eye = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye              # [src, tgt]
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)                       # nearer enemy -> heavier
    contrib = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
    return contrib.sum(dim=0)                                            # [P] summed over sources
def potential_attack_risk(obs, cache, *, horizon: float, player_id: int, config) -> Tensor:
    """Precautionary potential-attack RISK_SCORE per planet — ``[P]``.
    Unlike :func:`cheap_enemy_pressure` (reachable enemy *garrison* mass), this is a
    *precaution* map built from enemy **planets** as potential attackers. For every
    ``(enemy planet e, planet t)`` pair it adds a threat term that grows with the
    enemy's strength (ships + production) and proximity (≈ low straight-line arrival
    time), but is **zeroed when the sun blocks the direct path** ``e→t`` — the engine
    kills fleets whose segment grazes the sun, so such an enemy cannot attack ``t``
    directly (e.g. opposite corners across the centre). RISK(t) is the aggregate over
    all enemy planets, **discounted by friendly support** (nearby owned planets and
    their ships, which can reinforce), and scaled by the planet's own value
    (production). Added into the regroup gradient so leftover ships flow from low-risk
    toward high-risk owned planets.
    Pure arithmetic on cached tensors (one ``P×P`` pass) — CPU/CUDA agree, no host sync.
    The sun gate is applied to the enemy threat only (reinforcement may route around
    the sun over the horizon, so friendly support is left ungated by design).
    """
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    H = max(float(horizon), 1e-6)
    d0 = cache.cross_dist[0].to(dtype)                                   # [src, tgt] current centre dist
    ships = obs.ships.to(dtype)
    prod = obs.prod.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))                          # [P] per-source fleet speed
    reach = (speeds.view(P, 1) * H).clamp(min=1e-6)                      # [src, 1] straight-line reach in H
    decay = (1.0 - d0 / reach).clamp(min=0.0)                            # [src, tgt] nearer/faster -> heavier
    eye = torch.eye(P, device=device, dtype=torch.bool)
    # --- sun line-of-sight gate: 0 where segment src->tgt grazes the sun --------
    x = obs.x.to(dtype)
    y = obs.y.to(dtype)
    ax = x.view(P, 1); ay = y.view(P, 1)                                 # src endpoint  [src,1]
    bx = x.view(1, P); by = y.view(1, P)                                 # tgt endpoint  [1,tgt]
    abx = bx - ax; aby = by - ay                                         # [src, tgt]
    denom = (abx * abx + aby * aby).clamp(min=1e-12)
    u = (((CENTER - ax) * abx + (CENTER - ay) * aby) / denom).clamp(0.0, 1.0)
    cx = ax + u * abx; cy = ay + u * aby                                 # closest point on segment to centre
    sun_dist = torch.sqrt(((cx - CENTER) ** 2 + (cy - CENTER) ** 2).clamp(min=0.0))
    los_clear = (sun_dist >= float(SUN_RADIUS)).to(dtype)               # [src, tgt] 1 = unobstructed
    # --- aggregated enemy-planet threat on each target -------------------------
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))  # [P]
    strength = ships + float(config.risk_enemy_prod_weight) * prod      # [P] per enemy source
    valid_e = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye            # [src, tgt]
    threat = torch.where(valid_e, strength.view(P, 1) * decay * los_clear, torch.zeros_like(decay))
    enemy_threat = threat.sum(dim=0)                                    # [tgt] summed over enemy planets
    # --- friendly support discount (own neighbours + their ships) --------------
    own = obs.owned & obs.alive                                         # [P]
    valid_o = own.view(P, 1) & obs.alive.view(1, P) & ~eye             # [neighbour, tgt]
    support = torch.where(
        valid_o, (1.0 + ships).view(P, 1) * decay, torch.zeros_like(decay)
    ).sum(dim=0)                                                       # [tgt]
    value = 1.0 + float(config.risk_self_prod_weight) * prod           # [tgt] own-planet worth
    return value * enemy_threat / (1.0 + float(config.risk_support_weight) * support)  # [P]

# ===========================================================================
# PPO checkpoint update 0400 reranker (embedded weights)
# ===========================================================================
_PPO_W1_DATA = [[-0.07937896251678467, -0.04291972890496254, -0.0740223228931427, -0.12120762467384338, 0.023120354861021042, 0.05084563419222832, 0.07100701332092285, -0.1374155580997467, 0.15981854498386383, -0.17559927701950073, -0.08043313026428223, 0.005581718869507313, 0.061441414058208466, 0.1242639422416687, 0.1929267793893814, -0.1700231432914734, 0.06851652264595032, 0.043279215693473816, 0.17817533016204834, 0.16108235716819763, 0.014202137477695942, -0.0178989265114069, -0.04914389178156853, 0.06588190048933029, 0.39433538913726807, 0.043888114392757416, 0.04278544709086418, -0.09598100930452347, 0.012907855212688446, 0.056857287883758545, 0.30125191807746887, 0.08687601238489151, 0.17454324662685394, -0.021777324378490448, 0.16236211359500885, 0.05112011730670929, 0.01354168076068163, -0.07837338745594025, -0.15076568722724915, 0.12092746794223785, 0.0710395872592926, -0.11302967369556427, 0.06927403062582016, 0.25619080662727356, 0.14114058017730713, 0.05414637178182602, 0.03234585374593735, -0.1666632890701294, 0.11202718317508698, 0.00841248594224453, -0.25882089138031006, 0.4187869727611542, -0.05124253034591675, 0.19128036499023438, 0.2871084213256836, 0.10550527274608612, 0.1328614056110382, -0.0780089870095253, 0.17718642950057983, -0.21191804111003876, 0.3764401376247406, 0.0535568930208683, -0.08700501173734665, -0.0036202173214405775, -0.2527006268501282, -0.09492912143468857, -0.09851311147212982, 0.1985035091638565, -0.06963823735713959, -0.27124306559562683, -0.052372101694345474, -0.17711056768894196, 0.06875962018966675, 0.06732860207557678, 0.09413876384496689, -0.10001856088638306, -0.17329660058021545, 0.06290076673030853, -0.03918212652206421, -0.06138104945421219, 0.02693423628807068, -0.033777303993701935, 0.07861112803220749, -0.25096309185028076, 0.06383214890956879, 0.023649943992495537, -0.02451718971133232, -0.0021496980916708708, 0.09379015117883682, -0.08791526407003403, 0.0010227722814306617, -0.08174047619104385, -0.06200278177857399, 0.05506538599729538, 0.28713706135749817, -0.06579940766096115], [-0.129670649766922, 0.06604216992855072, -0.15591216087341309, 0.08650226145982742, -0.04688990116119385, 0.07740895450115204, 0.01840093359351158, 0.0015222904039546847, -0.12868519127368927, 0.09627284109592438, 0.01908859983086586, -0.06907688826322556, 0.008220404386520386, -0.11017435789108276, -0.1231703832745552, 0.17502272129058838, 0.09561771154403687, 0.11521261185407639, -0.10870540142059326, -0.10693749040365219, 0.024350758641958237, 0.12497011572122574, -0.025782698765397072, -0.08202775567770004, 0.2426508218050003, 0.11491814255714417, -0.10787069797515869, 0.0999886766076088, 0.0556613989174366, 0.04341552034020424, -0.03267425298690796, -0.06321847438812256, -0.19458350539207458, -0.12963664531707764, -0.11781314760446548, 0.06022138148546219, 0.0762922614812851, 0.11680848896503448, 0.08143706619739532, -0.02348850853741169, -0.014871180057525635, -0.1671895831823349, 0.011229395866394043, -0.12887650728225708, 0.2726380527019501, 0.0839848443865776, -0.07698006182909012, 0.12110456079244614, -0.09588862210512161, -0.02974746935069561, -0.15800443291664124, -0.1348862498998642, 0.014119183644652367, -0.11849111318588257, -0.12813012301921844, -0.09847152978181839, -0.07242514938116074, 0.02281116135418415, -0.16430716216564178, 0.119920514523983, 0.25415661931037903, 0.10249104350805283, 0.06518128514289856, -0.04913414269685745, 0.2027578502893448, 0.02372886799275875, 0.08031316846609116, -0.1199805736541748, -0.10694894939661026, 0.14175592362880707, 0.09002310782670975, 0.1473391205072403, -0.017580995336174965, 0.058760400861501694, -0.04190574958920479, 0.044875841587781906, 0.10785750299692154, -0.006991113536059856, -0.12865176796913147, 0.024295557290315628, 0.0922498032450676, -0.0499073788523674, -0.02653665468096733, 0.06111515685915947, -0.0452452227473259, -0.1276080459356308, 0.047998394817113876, 0.2409900277853012, -0.018265968188643456, 0.020184095948934555, 0.13502301275730133, 0.055793747305870056, 0.006932158954441547, 0.024193817749619484, -0.06763579696416855, 0.026910830289125443], [1.0991564989089966, 0.04049328342080116, 0.05941658094525337, 0.03715888410806656, 0.010052141733467579, -0.04515888914465904, -0.0536273792386055, -1.0253938436508179, -0.0495043620467186, 0.004367919638752937, 0.08802637457847595, -0.03220871090888977, -0.03454481437802315, -0.06776309758424759, -0.03334183245897293, 0.0322730615735054, -0.05474785715341568, -0.032396890223026276, 0.06944586336612701, 0.06647517532110214, 0.024134129285812378, 0.006764194928109646, 0.026719439774751663, -0.05169102922081947, 0.7806380391120911, -0.03496187552809715, -0.02876828797161579, 0.016845813021063805, 0.028719954192638397, -0.045366279780864716, 0.07964863628149033, -0.018075890839099884, -0.0255866851657629, 0.01863180287182331, -0.035561393946409225, -0.038639459758996964, -0.00602986803278327, 0.05062992870807648, 0.07304168492555618, 0.018466375768184662, -0.0815052017569542, 1.5707228183746338, -0.041152793914079666, 0.0026821913197636604, -1.51751708984375, -0.03863482549786568, -0.03264988586306572, 0.010433314368128777, -0.07736586034297943, -0.02829998917877674, -0.6780509948730469, 0.1977144181728363, 0.012029854580760002, 0.04669521749019623, 0.08979535847902298, 0.05646098777651787, 0.03539483621716499, -0.02483253739774227, -0.01712949573993683, -0.03423626348376274, 0.7576169967651367, -0.04256176948547363, 0.03651576116681099, -0.020387176424264908, 0.06635557115077972, 0.08405690640211105, 0.024884480983018875, 0.0048739127814769745, 0.06302822381258011, -0.08970939368009567, 0.042361605912446976, 0.008511633612215519, -0.05082376301288605, -0.08428049832582474, 0.027020350098609924, 0.05613104999065399, 0.031447526067495346, -0.08657458424568176, 0.9316509366035461, 0.04105589911341667, -0.01681399904191494, 0.01947205327451229, 0.036605823785066605, -0.07361659407615662, -0.03918220475316048, -0.019623257219791412, 0.09012670814990997, 0.009454316459596157, 0.051379602402448654, 0.039377737790346146, -0.00870151910930872, 0.03523936867713928, 0.033947285264730453, -0.032155461609363556, 0.08562654256820679, 0.03738528490066528], [0.04998151957988739, 0.23322796821594238, -0.4125191569328308, -0.05624183639883995, 0.1236148253083229, 0.2711959779262543, 0.23114582896232605, 0.19556711614131927, 0.08290226757526398, 0.1924780011177063, -0.14166681468486786, -0.0589546374976635, 0.12035910040140152, 0.11710388958454132, 0.048790764063596725, -0.07401066273450851, 0.34877097606658936, 0.29299822449684143, -0.018097160384058952, -0.007312926929444075, 0.09785407036542892, 0.4185692071914673, -0.16571080684661865, -0.3095852732658386, -0.6961370706558228, 0.38290712237358093, -0.39662718772888184, -0.05586635693907738, 0.0910533145070076, 0.2051704227924347, -0.1640327274799347, 0.07175807654857635, 0.08109171688556671, -0.42919856309890747, 0.07491607218980789, 0.21663238108158112, 0.24046780169010162, 0.4763687551021576, -0.10774674266576767, -0.18549472093582153, 0.1781194508075714, 0.13430270552635193, 0.21005934476852417, 0.03148003667593002, -0.12291596084833145, 0.2886662781238556, -0.23894546926021576, -0.037793319672346115, 0.10243244469165802, -0.1249602809548378, 0.38034528493881226, -0.4400390684604645, -0.11274295300245285, -0.002792724873870611, -0.010674095712602139, 0.005436865147203207, 0.0012424200540408492, 0.01603267900645733, 0.055968109518289566, -0.011786133050918579, -0.6462038159370422, 0.30722522735595703, -0.0806010290980339, -0.1283530443906784, -0.11814385652542114, -0.16577164828777313, -0.08368412405252457, 0.011792573146522045, -0.25620517134666443, 0.012028233148157597, 0.3715969920158386, -0.03370971232652664, 0.2040969282388687, 0.20740742981433868, -0.09181929379701614, -0.13930858671665192, -0.05708873271942139, 0.10015605390071869, -0.03991267457604408, -0.12178201228380203, 0.31901732087135315, -0.27999410033226013, 0.019412605091929436, 0.01902652718126774, 0.17476044595241547, -0.40836992859840393, 0.07424478232860565, 0.35899049043655396, 0.005312069319188595, -0.13502603769302368, 0.42136529088020325, -0.12320944666862488, -0.14842589199543, 0.1575271487236023, -0.024547968059778214, -0.1576494425535202], [0.07903766632080078, 0.3773055076599121, -0.7576808333396912, -0.33136382699012756, 0.3892078697681427, 0.5647314190864563, 0.5334287881851196, -0.05026596784591675, 0.3149016499519348, 0.022844403982162476, -0.5057504773139954, -0.29988735914230347, 0.4369453489780426, 0.4723074436187744, 0.17553067207336426, -0.2780468463897705, 0.6432520747184753, 0.5516549348831177, 0.024034209549427032, 0.1750929057598114, 0.3424370288848877, 0.49923110008239746, -0.46235811710357666, -0.43070682883262634, -0.11352240294218063, 0.5889014601707458, -0.47633886337280273, -0.33344200253486633, 0.2980973422527313, 0.5142097473144531, -0.020869314670562744, 0.35312607884407043, 0.33412742614746094, -0.5877423286437988, 0.3188208043575287, 0.5195744037628174, 0.4490986764431, 0.4578641653060913, -0.4176923334598541, 0.0004216572851873934, 0.4999563992023468, 0.10272688418626785, 0.5200905203819275, -0.030274640768766403, -0.10470753163099289, 0.5703745484352112, -0.3894083797931671, -0.19925296306610107, 0.4192385971546173, -0.3630068898200989, -0.03398335725069046, -0.06207866221666336, -0.39744359254837036, 0.0399334691464901, -0.04344292730093002, 0.23020566999912262, 0.19187965989112854, -0.233529195189476, 0.20633089542388916, -0.04664744809269905, -0.12459307909011841, 0.5503001809120178, -0.3914489150047302, -0.3834269046783447, -0.32951146364212036, -0.5276232361793518, -0.4024350047111511, 0.10507531464099884, -0.5967485308647156, 0.02578393556177616, 0.4621390402317047, -0.15728066861629486, 0.538912832736969, 0.5356612205505371, 0.10494980961084366, -0.45601755380630493, -0.21584555506706238, 0.43432581424713135, 0.026176758110523224, -0.43701043725013733, 0.5450782179832458, -0.5295447707176208, 0.24812056124210358, 0.047028325498104095, 0.4862045347690582, -0.4928451180458069, 0.2681715488433838, 0.5248310565948486, 0.22602392733097076, -0.44853755831718445, 0.549169659614563, -0.44333040714263916, -0.45940980315208435, 0.4637143015861511, -0.06957665830850601, -0.4864414632320404], [0.28010258078575134, 0.11505692452192307, -0.09353417903184891, -0.10394590348005295, 0.13011406362056732, 0.04492306709289551, 0.06366857886314392, -0.30454546213150024, 0.02064242959022522, -0.26945266127586365, -0.03425150364637375, -0.23118865489959717, 0.08095911890268326, 0.010863825678825378, 0.01884145475924015, -0.061493221670389175, 0.07708843052387238, 0.09297963231801987, 0.07190057635307312, -0.006218662019819021, 0.11727941781282425, 0.10641124844551086, -0.13800936937332153, -0.1072305217385292, -0.08796995133161545, 0.07521339505910873, -0.101545549929142, -0.12757432460784912, 0.1515170931816101, 0.09184494614601135, -0.0641016811132431, 0.06046866253018379, -0.004596401005983353, -0.09971359372138977, 0.01873200200498104, 0.0688851848244667, 0.09879976511001587, 0.12315669655799866, -0.032480645924806595, 0.4483172595500946, -0.003999405540525913, 0.25184762477874756, 0.07030648738145828, 0.016434866935014725, -0.23805411159992218, 0.06768041849136353, -0.12917830049991608, -0.004847974982112646, 0.045975346118211746, -0.1361890584230423, -0.18660719692707062, -0.48344671726226807, -0.07493730634450912, 0.022284550592303276, -0.060037944465875626, 0.1045503243803978, 0.12253820896148682, -0.1755501925945282, -0.02123752050101757, -0.0261207465082407, 0.014532288536429405, 0.05979803204536438, -0.06965971738100052, -0.11628580093383789, 0.0703447088599205, 0.014438414946198463, -0.024285046383738518, 0.04437248036265373, -0.05404463782906532, 0.030041450634598732, 0.11000274866819382, -0.029718060046434402, 0.04421757534146309, 0.030456138774752617, 0.3094911575317383, -0.028876306489109993, -0.031311750411987305, 0.08832883834838867, 0.3250594139099121, -0.058225199580192566, 0.1262417435646057, -0.08497145026922226, 0.11753115803003311, -0.044220611453056335, 0.05221984162926674, -0.13279692828655243, 0.25961005687713623, 0.23154497146606445, 0.07898904383182526, -0.024935007095336914, 0.08402469009160995, -0.04331983998417854, -0.06112787872552872, 0.08430103212594986, 0.0017348737455904484, -0.02947513572871685], [0.21695010364055634, 0.03052198514342308, -0.010584627278149128, -0.07434895634651184, 0.045894138514995575, -0.04402687028050423, -0.052365317940711975, -0.21099916100502014, -0.07301788032054901, -0.23527181148529053, 0.006386978551745415, -0.13974909484386444, 0.07164442539215088, -0.05522448942065239, -0.07599637657403946, 0.05323358252644539, -0.04145065322518349, 0.0056911068968474865, -0.02439327910542488, -0.07089781016111374, 0.025282850489020348, 0.03279483690857887, -0.05740208551287651, -0.06725169718265533, -0.21367797255516052, 0.006639879196882248, -0.044048093259334564, -0.07799886912107468, 0.14448314905166626, 0.04230278357863426, -0.1868172138929367, 0.03641586750745773, -0.09712114185094833, 0.027526460587978363, -0.09130024909973145, -0.006414642091840506, 0.017142776399850845, 0.01627102866768837, 0.045619718730449677, 0.3787369132041931, -0.0667726919054985, 0.1771031618118286, 0.008640848100185394, -0.12383569031953812, -0.1694476306438446, 0.029099684208631516, -0.05939262732863426, 0.040985967963933945, -0.004794662818312645, -0.05000011995434761, -0.10761529952287674, -0.6155959367752075, -0.0529010146856308, -0.04009132832288742, -0.16756103932857513, 0.03293484449386597, 0.015087183564901352, -0.10931396484375, -0.057098545134067535, 0.08384726196527481, -0.17544867098331451, -0.01600506715476513, 0.004939054138958454, -0.05523977056145668, 0.13589929044246674, 0.05554412677884102, 0.017741667106747627, -0.008913464844226837, 0.013702310621738434, 0.19639411568641663, -0.020024485886096954, 0.03894335404038429, -0.043024517595767975, -0.004931148607283831, 0.20347845554351807, 0.020382840186357498, 0.03159044310450554, 0.0012941324384883046, 0.2042670100927353, -0.006150552071630955, 0.04369530454277992, -0.007975134998559952, 0.06894411146640778, 0.06920699775218964, -0.033844929188489914, -0.07209711521863937, 0.1933119297027588, 0.14632895588874817, 0.06839796900749207, 0.027759362012147903, 0.01914338394999504, 0.015784695744514465, 0.0008550774655304849, -0.003696058876812458, -0.13359443843364716, -0.0044230325147509575], [-0.15426293015480042, -0.021588154137134552, 0.014675602316856384, 0.04801437258720398, 0.016508260741829872, -0.021437078714370728, -0.03657032921910286, 0.2079716920852661, -0.08717051893472672, -0.005932745989412069, 0.03416702151298523, -0.016523882746696472, 0.07021486759185791, -0.01885860785841942, -0.17690622806549072, 0.07605694234371185, -0.0018936217529699206, 0.03379754722118378, -0.19442865252494812, -0.16426196694374084, -0.008509705774486065, 0.013514477759599686, -0.07960566133260727, -0.0021975315175950527, -0.4672454297542572, 0.03965374082326889, 0.03347165510058403, -0.02183438092470169, 0.17029495537281036, 0.01650920696556568, -0.4942285716533661, 0.0049106585793197155, -0.04699316620826721, 0.021934201940894127, -0.12061707675457001, -0.048426516354084015, 0.025960959494113922, -0.024718396365642548, 0.05615793541073799, 0.07186152040958405, -0.060332685708999634, -0.3042212724685669, 0.0022741376888006926, -0.3496866822242737, 0.2786969840526581, 0.01806659623980522, 0.003563710954040289, 0.150616854429245, -0.065839484333992, -0.004764575976878405, 0.34038880467414856, -0.647620677947998, -0.03919650986790657, -0.1845656931400299, -0.3945091664791107, -0.006672232877463102, -0.09492641687393188, 0.01667013019323349, -0.1192936971783638, 0.2311072051525116, -0.44851964712142944, -0.006930461153388023, 0.05348135903477669, 0.008581326343119144, 0.13508914411067963, 0.09052518010139465, 0.02377130277454853, -0.23581655323505402, 0.021778110414743423, 0.4004722237586975, 0.019184451550245285, 0.1766214370727539, -0.04207745939493179, -0.01795685850083828, -0.01656903699040413, -0.009121442213654518, 0.13214626908302307, -0.033746007829904556, -0.09831544011831284, 0.008285355754196644, 0.025566160678863525, -0.00838469062000513, 0.013857431709766388, 0.3028917908668518, 0.002616725629195571, -0.035220883786678314, 0.012742247432470322, 0.13430307805538177, -0.0419842004776001, -0.005880087148398161, -6.373241922119632e-05, -0.033697038888931274, -0.0007910849526524544, -0.006662081927061081, -0.3403986394405365, 0.056644000113010406], [0.006766696460545063, -0.017604852095246315, 0.02261820062994957, 0.010756488889455795, -0.03623132035136223, -0.017702613025903702, 0.014798938296735287, 0.1793365180492401, -0.02140333130955696, 0.059783171862363815, -0.001164700835943222, 0.0075800372287631035, -0.003511647693812847, 0.04864038527011871, -0.008415587246418, 0.014481068588793278, -0.023652736097574234, 0.012084590271115303, -0.09435790777206421, 0.015361521393060684, -0.047112081199884415, 0.007271166890859604, -0.023403020575642586, 0.025209423154592514, -0.37290245294570923, -0.006460423115640879, 0.0018699197098612785, 0.024156952276825905, 0.007349467370659113, -0.00959936622530222, -0.19143062829971313, -0.012664594687521458, 0.05323176085948944, -0.00035215550451539457, 0.022190075367689133, -0.01920050010085106, -0.0067905643954873085, -0.025005998089909554, -0.003287803614512086, 0.027657845988869667, 0.00833206158131361, -0.07053586095571518, 0.02336745895445347, -0.03871699795126915, 0.08353082090616226, -0.010912350378930569, 0.008275067433714867, 0.020386146381497383, -0.02749413624405861, 0.002289365977048874, 0.2550801932811737, -0.05936624854803085, -0.019663764163851738, -0.08344311267137527, -0.04945066198706627, -0.03744423761963844, -0.05771666765213013, 0.028737211599946022, -0.01897510141134262, 0.03368861973285675, -0.3935590088367462, 0.00658809207379818, 0.02755039930343628, 0.03635244444012642, 0.004057020880281925, -0.07007085531949997, -0.02190808206796646, -0.09726285189390182, -0.038381222635507584, 0.011042732745409012, 0.0012670758878812194, 0.001512890332378447, -0.01037430576980114, 0.028233688324689865, -0.05123355612158775, -0.01209894847124815, -0.006104040890932083, -0.04221327602863312, -0.007730844430625439, 0.034516070038080215, 0.011419996619224548, 0.008556264452636242, -0.02003912813961506, 0.1357060968875885, -0.0007727438933216035, -0.030068350955843925, 0.006244445685297251, 0.08130011707544327, -0.054185230284929276, -0.007872452959418297, -0.016702478751540184, -0.003904282581061125, 0.008826944045722485, 0.0179741270840168, -0.11674510687589645, -0.0033380307722836733], [-0.05020768195390701, 0.02621695026755333, -0.2902998924255371, 0.13358783721923828, 0.0438217967748642, 0.1369418054819107, 0.14307402074337006, -0.01741781458258629, 0.0034788348712027073, -0.007443603128194809, -0.02183583937585354, -0.04154060781002045, -0.008353649638593197, -0.04584212973713875, -0.12735798954963684, 0.06329097598791122, 0.20194277167320251, 0.09724704921245575, -0.05222458019852638, -0.15604649484157562, -0.030865781009197235, 0.1673533171415329, -0.08334401249885559, -0.0274492334574461, 0.00458585424348712, 0.1390678882598877, -0.09431468695402145, 0.07305227220058441, -0.01430633757263422, 0.09499729424715042, -0.01660127192735672, -0.04477524757385254, -0.030995050445199013, -0.08930490165948868, -0.06167387589812279, 0.07797621190547943, 0.03479449823498726, 0.12348824739456177, 0.028112735599279404, -0.028610626235604286, 0.14201883971691132, -0.06839489191770554, 0.11812061071395874, -0.3234880864620209, 0.0872810110449791, 0.13922858238220215, -0.028246840462088585, 0.09395141154527664, -0.05777807906270027, -0.0434066504240036, -0.024285655468702316, -0.14426764845848083, 0.015491865575313568, -0.10413189977407455, -0.3032529056072235, -0.09581159055233002, -0.02653103694319725, -0.02974672056734562, -0.03599826991558075, 0.21287289261817932, 0.011952269822359085, 0.14525386691093445, -0.02315143309533596, -0.03227877616882324, 0.0675242617726326, -0.06970872730016708, 0.006310836877673864, -0.12942972779273987, -0.2140427827835083, 0.2573167085647583, -0.01735510863363743, 0.11456079035997391, 0.12191528081893921, 0.18453159928321838, -0.0384102389216423, -0.034274693578481674, 0.14296749234199524, -0.003804328152909875, -0.07046942412853241, -0.037723056972026825, 0.16910488903522491, -0.12008217722177505, -0.07825671881437302, 0.13372454047203064, 0.0792548805475235, -0.09981884062290192, 0.03705257549881935, 0.1041550561785698, -0.07752273231744766, -0.00015230069402605295, 0.16867122054100037, 0.011232822202146053, -0.03872446343302727, 0.02799963764846325, -0.16047224402427673, -0.10328397154808044], [-0.08804628998041153, -0.03366412967443466, 0.15619128942489624, 0.024574439972639084, -0.05996876582503319, -0.08117608726024628, -0.11631719022989273, 0.0903606116771698, -0.11840912699699402, -0.020332643762230873, 0.10517382621765137, 0.06554252654314041, -0.06825532764196396, -0.1141030415892601, -0.07030091434717178, 0.12248308211565018, -0.13597579300403595, -0.060607749968767166, -0.04275645688176155, -0.0353383794426918, -0.056949615478515625, -0.04304370656609535, 0.07947620004415512, 0.003763932967558503, -0.0002652498660609126, -0.07216113060712814, 0.0488145686686039, 0.07141950726509094, 0.005584362428635359, -0.0868721455335617, -0.10364893823862076, -0.03849789872765541, -0.11086247116327286, 0.042522139847278595, -0.11770391464233398, -0.09412972629070282, -0.05215537175536156, -0.02932475507259369, 0.08117678016424179, -0.010963725857436657, -0.11914706230163574, -0.1725834310054779, -0.13375568389892578, 0.012954406440258026, 0.17587047815322876, -0.10296636819839478, 0.03763670474290848, 0.05932140722870827, -0.09948360174894333, 0.05685121566057205, -0.021916821599006653, -0.303541362285614, 0.07340097427368164, -0.04423408582806587, -0.018342141062021255, 0.0035083245020359755, -0.016310187056660652, 0.035978663712739944, -0.14177724719047546, -0.004597794264554977, -0.011850402690470219, -0.07870037108659744, 0.09108652919530869, 0.03592675179243088, 0.19375813007354736, 0.14942273497581482, 0.09435046464204788, -0.02150505781173706, 0.12787388265132904, 0.03996555134654045, -0.03524263575673103, 0.06341902166604996, -0.12681815028190613, -0.09198372066020966, -0.03320712968707085, 0.10949640721082687, 0.031614530831575394, -0.03995981439948082, -0.05332879722118378, 0.07606805115938187, -0.08685238659381866, 0.07705625891685486, -0.015728261321783066, 0.017378730699419975, -0.08867083489894867, 0.04919307306408882, -0.02218685857951641, -0.10625319182872772, -0.011555629782378674, 0.07688111066818237, -0.053874820470809937, 0.08336144685745239, 0.0844787210226059, -0.05575534328818321, -0.014125513844192028, 0.09682736545801163], [0.27772843837738037, -0.10888146609067917, 0.13221284747123718, 0.3684788942337036, -0.19758804142475128, -0.14370495080947876, -0.1460038721561432, -0.23869487643241882, -0.1991172879934311, -0.5737242102622986, 0.19418293237686157, -0.057659946382045746, -0.24375291168689728, -0.1882469207048416, -0.308796226978302, 0.24683429300785065, -0.1240072101354599, -0.1795685738325119, -0.3761671483516693, -0.41517019271850586, -0.16907307505607605, -0.19279178977012634, 0.21534959971904755, 0.15122437477111816, 0.1690288782119751, -0.181549534201622, 0.1432822346687317, 0.31313204765319824, -0.31852391362190247, -0.18331794440746307, 0.08024252206087112, -0.2750266492366791, -0.22335034608840942, 0.17154738306999207, -0.20321710407733917, -0.1393866240978241, -0.19735881686210632, -0.14944633841514587, 0.24118059873580933, 0.5510922074317932, -0.12815771996974945, 0.18198563158512115, -0.12404462695121765, -0.5277982950210571, -0.27005642652511597, -0.14989084005355835, 0.14886870980262756, 0.34056970477104187, -0.23172414302825928, 0.0850287675857544, -0.1556273102760315, -0.2160956859588623, 0.2066670060157776, -0.41266945004463196, -0.5221652984619141, -0.44416606426239014, -0.3855902850627899, 0.06668952852487564, -0.2348504513502121, 0.49206456542015076, 0.15595312416553497, -0.1427987515926361, 0.16895602643489838, 0.16317057609558105, 0.20333409309387207, 0.09973002970218658, 0.16556459665298462, -0.43826350569725037, 0.1112939640879631, 0.455960750579834, -0.1321043223142624, 0.3404214382171631, -0.13116057217121124, -0.16947321593761444, 0.29604771733283997, 0.17296946048736572, 0.3679695725440979, -0.26063600182533264, 0.40418967604637146, 0.1838465929031372, -0.18935056030750275, 0.16522164642810822, -0.3735558092594147, 0.4154054820537567, -0.16383129358291626, 0.19191662967205048, 0.22059333324432373, -0.30759739875793457, -0.3985956609249115, 0.19065004587173462, -0.1919507086277008, 0.2060612440109253, 0.16642862558364868, -0.16860195994377136, -0.42287158966064453, 0.1518888920545578]]
_PPO_B1_DATA = [-0.25226038694381714, -0.024942422285676003, -0.062126439064741135, 0.584811806678772, -0.15150046348571777, -0.00116110616363585, -0.11865728348493576, 0.32785654067993164, -0.44136664271354675, -0.002650071633979678, 0.3178965747356415, 0.05782102793455124, -0.24960017204284668, -0.44603559374809265, -0.6071462035179138, 0.5775089263916016, 0.0021370479371398687, -0.033373020589351654, -0.6204684376716614, -0.667141318321228, -0.17220228910446167, 0.05294889956712723, 0.11806873232126236, 0.016643956303596497, -0.3850381672382355, 0.026023969054222107, -0.022363174706697464, 0.4910326302051544, -0.2653729021549225, -0.11145319789648056, -0.4707285463809967, -0.44999998807907104, -0.5811559557914734, -0.020162217319011688, -0.515449047088623, -0.07525798678398132, -0.022338518872857094, 0.04651010408997536, 0.5203443765640259, 0.0692257434129715, -0.18914775550365448, -0.46693769097328186, -0.16527174413204193, -0.7499770522117615, 0.3846920132637024, -0.03686295822262764, 0.03856603801250458, 0.5840703248977661, -0.42675718665122986, 0.0982021689414978, 0.37277182936668396, -0.6671375036239624, 0.2228001058101654, -0.6286720037460327, -0.7517541646957397, -0.5847491025924683, -0.5760279893875122, 0.3636102080345154, -0.5255795121192932, 0.7343183159828186, -0.3826286494731903, -0.03190920129418373, 0.2946758568286896, 0.11975748836994171, 0.5464974045753479, 0.346161812543869, 0.38105106353759766, -0.6319102048873901, 0.042823441326618195, 0.7563364505767822, -0.01986583322286606, 0.6122007966041565, -0.13036733865737915, -0.13266189396381378, -0.06382523477077484, 0.3770313858985901, 0.6534557938575745, -0.297212690114975, -0.11872801929712296, 0.18512442708015442, 0.029537247493863106, 0.034099280834198, -0.4937582015991211, 0.6978296041488647, -0.17458118498325348, -0.006719937082380056, 0.061244070529937744, 0.07840806245803833, -0.5433229207992554, 0.346987247467041, 0.08038289099931717, 0.32271623611450195, 0.20582027733325958, -0.21562695503234863, -0.7475572824478149, 0.14440016448497772]
_PPO_W2_DATA = [[-0.5586916208267212], [-0.2164461463689804], [0.27964723110198975], [0.24315094947814941], [-0.1897512823343277], [-0.23117808997631073], [-0.26631805300712585], [0.4203971326351166], [-0.24891197681427002], [0.2815852463245392], [0.24869753420352936], [0.15162821114063263], [-0.2163321077823639], [-0.25789716839790344], [-0.2624756097793579], [0.26327040791511536], [-0.2963891923427582], [-0.21729935705661774], [-0.2997443377971649], [-0.3299180567264557], [-0.1644650101661682], [-0.2411976456642151], [0.21488907933235168], [0.24304112792015076], [-0.540199339389801], [-0.25409314036369324], [0.26811546087265015], [0.2223622351884842], [-0.16854393482208252], [-0.20555156469345093], [-0.3253548741340637], [-0.22140975296497345], [-0.25813552737236023], [0.23625324666500092], [-0.26022908091545105], [-0.21637539565563202], [-0.20526014268398285], [-0.30737942457199097], [0.2532195448875427], [-0.2694077789783478], [-0.25998830795288086], [-0.9274510741233826], [-0.21418382227420807], [-0.4176381528377533], [0.8812236189842224], [-0.26775673031806946], [0.22430650889873505], [0.2626226842403412], [-0.22552086412906647], [0.17783765494823456], [0.3283548653125763], [-0.6607818603515625], [0.15833476185798645], [-0.30901870131492615], [-0.47080564498901367], [-0.2757549583911896], [-0.2549198269844055], [0.165939062833786], [-0.24564045667648315], [0.3870088458061218], [-0.48511597514152527], [-0.22441314160823822], [0.19686701893806458], [0.18220113217830658], [0.3657362461090088], [0.2482311725616455], [0.22521919012069702], [-0.2940029799938202], [0.21457445621490479], [0.45681333541870117], [-0.2568269371986389], [0.2647194564342499], [-0.22670164704322815], [-0.235165536403656], [-0.18735259771347046], [0.2500307559967041], [0.26791101694107056], [-0.16735431551933289], [-0.377657413482666], [0.17149868607521057], [-0.24888812005519867], [0.22666795551776886], [-0.20930005609989166], [0.35080668330192566], [-0.21262650191783905], [0.26181527972221375], [-0.11866335570812225], [-0.28111493587493896], [-0.23138582706451416], [0.20893236994743347], [-0.2315942943096161], [0.1992007941007614], [0.19285444915294647], [-0.18022969365119934], [-0.41278618574142456], [0.21886864304542542]]
_PPO_B2_DATA = [-2.6454758644104004]
_PPO_WEIGHT_CACHE = {}

def _ppo_weights(device, dtype):
    key = (str(device), dtype)
    cached = _PPO_WEIGHT_CACHE.get(key)
    if cached is None:
        cached = (
            torch.tensor(_PPO_W1_DATA, dtype=dtype, device=device),
            torch.tensor(_PPO_B1_DATA, dtype=dtype, device=device),
            torch.tensor(_PPO_W2_DATA, dtype=dtype, device=device),
            torch.tensor(_PPO_B2_DATA, dtype=dtype, device=device),
        )
        _PPO_WEIGHT_CACHE[key] = cached
    return cached

def _ppo_candidate_bonus(features: Tensor, valid: Tensor, *, weight: float, center: bool = True) -> Tensor:
    if features.numel() == 0:
        return torch.zeros(features.shape[:-1], dtype=features.dtype, device=features.device)
    w1, b1, w2, b2 = _ppo_weights(features.device, features.dtype)
    learned = (torch.tanh(features @ w1 + b1) @ w2 + b2).squeeze(-1) * 2.0
    learned = torch.where(valid, learned, torch.zeros_like(learned))
    if center and bool(valid.any()):
        denom = valid.to(features.dtype).sum().clamp(min=1.0)
        mean = learned.sum() / denom
        learned = torch.where(valid, learned - mean, learned)
    return learned * float(weight)

def _ppo_candidate_features(
    *, score: Tensor, obs, cand_src: Tensor, cand_send: Tensor, cand_eta: Tensor,
    cand_active: Tensor, cand_tgt_slot: Tensor, cand_is_def: Tensor, cand_valid: Tensor,
) -> Tensor:
    C = int(score.shape[0])
    dtype = score.dtype
    device = score.device
    if C == 0:
        return torch.empty(0, 12, dtype=dtype, device=device)
    active_send = torch.where(cand_active, cand_send.to(dtype), torch.zeros_like(cand_send.to(dtype)))
    total_send = active_send.sum(dim=-1)
    active_count = cand_active.to(dtype).sum(dim=-1).clamp(min=1.0)
    max_eta = torch.where(cand_active, cand_eta.to(dtype), torch.zeros_like(cand_eta.to(dtype))).max(dim=-1).values
    min_eta = torch.where(cand_active, cand_eta.to(dtype), torch.full_like(cand_eta.to(dtype), 99.0)).min(dim=-1).values
    first_src = cand_src[:, 0].clamp(0, obs.P - 1)
    tgt = cand_tgt_slot.clamp(0, obs.P - 1)
    target_prod = obs.prod[tgt].to(dtype)
    target_ships = obs.ships[tgt].to(dtype)
    source_ships = obs.ships[first_src].to(dtype)
    owner = obs.owner_abs[tgt]
    neutral = (owner < 0).to(dtype)
    enemy = ((owner >= 0) & (owner != int(obs.player_id))).to(dtype)
    defense = cand_is_def.to(dtype)
    step = obs.step.to(dtype) if hasattr(obs, 'step') else torch.zeros((), dtype=dtype, device=device)
    step_term = torch.ones_like(score) * step / 500.0
    safe_score = torch.where(torch.isfinite(score), score, torch.zeros_like(score))
    feat = torch.stack([
        safe_score / 50.0,
        target_prod / 8.0,
        target_ships / 200.0,
        source_ships / 200.0,
        total_send / 200.0,
        min_eta / 20.0,
        max_eta / 20.0,
        active_count / 4.0,
        neutral,
        enemy,
        defense,
        step_term,
    ], dim=-1)
    return torch.where(cand_valid.unsqueeze(-1), feat, torch.zeros_like(feat))

def plan_lite_waves(
    *,
    movement: PlanetMovement,
    obs,
    obs_tensors: dict,
    cache,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
):
    """Single-size, single-source attack planner + regroup.
    Builds exactly one candidate per ``(source, target)`` shortlist pair — fleet
    size = the source's max garrison launch (``safe_drain``) — scores them with the
    exact competitive flow diff, and greedily fires the best wave per target up to
    ``max_waves_per_turn``. Returns the combined ``LaunchEntries`` (attack waves ++
    regroup).
    """
    P = obs.P
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)
    H_axis = int(garrison_status.ships.shape[-1])
    H = max(H_axis - 1, 0)
    K_eta = max(1, min(int(config.horizon), H))
    W = max(1, int(config.max_waves_per_turn))
    source_mask = obs.owned & obs.alive & (obs.ships >= float(config.min_ships_to_launch))
    if not bool(source_mask.any()):
        return _empty_entries(device, dtype)
    S_cap = max(1, min(int(config.max_sources_per_lane), P))
    source_idx, source_exists = _candidate_indices(obs.ships, source_mask, S_cap)
    target_idx, target_exists = build_target_shortlist(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
        player_count=int(player_count),
    )
    if not bool(target_exists.any()):
        return _empty_entries(device, dtype)
    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    target_is_mine = obs.owned[target_idx.clamp(0, P - 1)]                       # [T]
    source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)                # [S]
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain = safe_drain(
        garrison_status, source_idx=source_idx, source_ships=source_ships,
        H_eff=H_eff, player_id=pid,
    )                                                                            # [S]
    # Uniform reach cap = K_eta (= horizon).
    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)          # [T]
    floor = capture_floor(
        garrison_status, target_idx=target_idx, k_max=K_eta,
        capture_overhead=1.0, player_id=pid,
    )                                                                            # [T, K]
    K = int(floor.shape[-1])
    # --- single fleet size = the max garrison launch (safe_drain) ---------------
    # Engine needs integer ship counts; floor (never exceed what's available).
    sizes = drain.view(S, 1).expand(S, T).floor()                                # [S, T]
    # Strict-superset reachability precheck (always on): defers the body screen to
    # candidates that can physically reach the target in time.
    active = reachable_mask(
        movement, source_idx=source_idx, target_idx=target_idx,
        fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap,
    ).squeeze(-1)                                                                # [S, T]
    aim = intercept_angle(
        movement,
        source_idx.unsqueeze(1),                                                 # [S, 1]
        target_idx.unsqueeze(0),                                                 # [1, T]
        sizes,                                                                    # [S, T]
        active=active,
    )
    angle = aim["angle"]                                                         # [S, T]
    eta = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T))
    # Capture-floor gate at each fleet's arrival turn (defenders grow with k). The
    # single size must clear the defender it lands on (size >= floor_at_arr). Owned
    # targets have floor 1 (reinforcement), so any positive send clears.
    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)  # [S,T]
        floor_at_arr = floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    else:
        floor_at_arr = torch.ones(S, T, dtype=dtype, device=device)
    clears_floor = sizes >= floor_at_arr                                         # [S, T]
    src_neq_tgt = source_idx.view(S, 1) != target_idx.view(1, T)
    valid = (
        viable & clears_floor & (sizes >= 1.0) & src_neq_tgt
        & source_exists.view(S, 1) & target_exists.view(1, T)
    )                                                                            # [S, T]
    size_fracs = tuple(float(x) for x in getattr(config, "single_source_size_fracs", (1.0,)) if float(x) > 0.0)
    if not bool(getattr(config, "enable_single_source_size_variants", False)):
        size_fracs = (1.0,)
    if not size_fracs:
        size_fracs = (1.0,)
    size_fracs = tuple(dict.fromkeys(size_fracs))
    if len(size_fracs) == 1 and abs(size_fracs[0] - 1.0) < 1e-9:
        Gs = 1
        ss_sizes_g = sizes.unsqueeze(-1)
        ss_angle_g = angle.unsqueeze(-1)
        ss_eta_g = eta.unsqueeze(-1)
        ss_valid_g = valid.unsqueeze(-1)
    else:
        frac_t = torch.tensor(size_fracs, dtype=dtype, device=device).view(1, 1, -1)
        Gs = int(frac_t.shape[-1])
        ss_sizes_g = (drain.view(S, 1, 1) * frac_t).floor()
        ss_active_g = reachable_mask(
            movement, source_idx=source_idx, target_idx=target_idx,
            fleet_sizes=ss_sizes_g, eta_cap=eta_cap,
        )                                                                        # [S, T, Gs]
        ss_aim_g = intercept_angle(
            movement,
            source_idx.view(S, 1, 1),
            target_idx.view(1, T, 1),
            ss_sizes_g,
            active=ss_active_g,
        )
        ss_angle_g = ss_aim_g["angle"]
        ss_eta_g = ss_aim_g["eta"]
        ss_viable_g = ss_aim_g["viable"] & (ss_eta_g <= eta_cap.view(1, T, 1))
        if K > 0:
            ss_k_arr = (ss_eta_g.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
            ss_floor_at_arr = floor.view(1, T, K).expand(S, T, K).gather(-1, ss_k_arr)
        else:
            ss_floor_at_arr = torch.ones(S, T, Gs, dtype=dtype, device=device)
        ss_valid_g = (
            ss_viable_g & (ss_sizes_g >= ss_floor_at_arr) & (ss_sizes_g >= 1.0)
            & src_neq_tgt.unsqueeze(-1)
            & source_exists.view(S, 1, 1) & target_exists.view(1, T, 1)
        )
    if not bool(config.enable_focus_fire):
        # --- original: one single-source candidate per (source, target); L = 1 ----
        L = 1
        C = S * T * Gs
        cand_src = source_idx.view(S, 1, 1).expand(S, T, Gs).reshape(C, L)
        cand_tgt_slot = target_idx.view(1, T, 1).expand(S, T, Gs).reshape(C)
        cand_tgt_short = torch.arange(T, device=device).view(1, T, 1).expand(S, T, Gs).reshape(C)
        cand_send = torch.where(ss_valid_g, ss_sizes_g, torch.zeros_like(ss_sizes_g)).reshape(C, L)
        cand_angle = ss_angle_g.reshape(C, L)
        cand_eta = torch.where(ss_valid_g, ss_eta_g, torch.ones_like(ss_eta_g)).reshape(C, L)
        cand_active = ss_valid_g.reshape(C, L)
        cand_valid = ss_valid_g.reshape(C)
    else:
        # --- focus fire: single-source candidates (widened to L) ++ pooled, same-
        # step multi-source strikes that combine to clear strong targets no single
        # planet can take. Contributors sharing ceil(eta) land together, and the
        # flow-diff sums them vs the defender at that step. Strictly additive: the
        # single-source slot-0 scores are identical to the original path.
        L = max(1, int(config.max_strike_sources))
        ST = S * T * Gs
        ss_src = torch.zeros(ST, L, dtype=torch.long, device=device)
        ss_src[:, 0] = source_idx.view(S, 1, 1).expand(S, T, Gs).reshape(-1)
        ss_send = torch.zeros(ST, L, dtype=dtype, device=device)
        ss_send[:, 0] = torch.where(ss_valid_g, ss_sizes_g, torch.zeros_like(ss_sizes_g)).reshape(-1)
        ss_angle = torch.zeros(ST, L, dtype=dtype, device=device)
        ss_angle[:, 0] = ss_angle_g.reshape(-1)
        ss_eta = torch.ones(ST, L, dtype=dtype, device=device)
        ss_eta[:, 0] = torch.where(ss_valid_g, ss_eta_g, torch.ones_like(ss_eta_g)).reshape(-1)
        ss_active = torch.zeros(ST, L, dtype=torch.bool, device=device)
        ss_active[:, 0] = ss_valid_g.reshape(-1)
        ss_tgt_slot = target_idx.view(1, T, 1).expand(S, T, Gs).reshape(-1)
        ss_tgt_short = torch.arange(T, device=device).view(1, T, 1).expand(S, T, Gs).reshape(-1)
        ss_valid = ss_valid_g.reshape(-1)
        # Pooled strikes on offensive (non-owned) targets: group eligible sources by
        # arrival step, take the minimal drain-desc prefix (>=2) that clears the floor.
        eligible = (
            viable & (sizes >= 1.0) & src_neq_tgt
            & source_exists.view(S, 1) & target_exists.view(1, T)
        )                                                                        # [S, T] (no single-source clears_floor)
        step_arr = eta.clamp(min=1.0, max=float(K_eta)).ceil().long()            # [S, T] arrival step
        pooled = []   # list of (target_short_idx, source_rows[:j])
        if L >= 2 and K > 0:
            for t in range(T):
                if bool(target_is_mine[t]):
                    continue
                rows = torch.nonzero(eligible[:, t], as_tuple=False).flatten()
                if int(rows.numel()) < 2:
                    continue
                steps_t = step_arr[rows, t]
                for k in torch.unique(steps_t).tolist():
                    k = int(k)
                    if k < 1 or (k - 1) >= K:
                        continue
                    grp = rows[steps_t == k]
                    if int(grp.numel()) < 2:
                        continue
                    gd = sizes[grp, t]
                    order = torch.argsort(gd, descending=True, stable=True)
                    grp = grp[order]
                    csum = torch.cumsum(gd[order], dim=0)
                    need = floor[t, k - 1]
                    hit = torch.nonzero(csum >= need, as_tuple=False)
                    if int(hit.numel()) == 0:
                        continue                                                  # group can't clear even combined
                    j = int(hit[0].item()) + 1                                    # minimal sufficient prefix
                    if j < 2 or j > L:
                        continue                                                  # j==1 already covered single-source
                    pooled.append((t, grp[:j]))
        if pooled:
            C2 = len(pooled)
            p_src = torch.zeros(C2, L, dtype=torch.long, device=device)
            p_send = torch.zeros(C2, L, dtype=dtype, device=device)
            p_angle = torch.zeros(C2, L, dtype=dtype, device=device)
            p_eta = torch.ones(C2, L, dtype=dtype, device=device)
            p_active = torch.zeros(C2, L, dtype=torch.bool, device=device)
            p_tgt_slot = torch.zeros(C2, dtype=torch.long, device=device)
            p_tgt_short = torch.zeros(C2, dtype=torch.long, device=device)
            for i, (t, grp) in enumerate(pooled):
                j = int(grp.numel())
                p_src[i, :j] = source_idx[grp]
                p_send[i, :j] = sizes[grp, t]
                p_angle[i, :j] = angle[grp, t]
                p_eta[i, :j] = eta[grp, t]
                p_active[i, :j] = True
                p_tgt_slot[i] = target_idx[t]
                p_tgt_short[i] = t
            cand_src = torch.cat([ss_src, p_src], dim=0)
            cand_send = torch.cat([ss_send, p_send], dim=0)
            cand_angle = torch.cat([ss_angle, p_angle], dim=0)
            cand_eta = torch.cat([ss_eta, p_eta], dim=0)
            cand_active = torch.cat([ss_active, p_active], dim=0)
            cand_tgt_slot = torch.cat([ss_tgt_slot, p_tgt_slot], dim=0)
            cand_tgt_short = torch.cat([ss_tgt_short, p_tgt_short], dim=0)
            cand_valid = torch.cat(
                [ss_valid, torch.ones(C2, dtype=torch.bool, device=device)], dim=0
            )
        else:
            cand_src, cand_send, cand_angle = ss_src, ss_send, ss_angle
            cand_eta, cand_active = ss_eta, ss_active
            cand_tgt_slot, cand_tgt_short, cand_valid = ss_tgt_slot, ss_tgt_short, ss_valid
        C = int(cand_src.shape[0])
    cand_is_def = target_is_mine[cand_tgt_short]                                  # [C]
    launches = make_launch_set(
        source_slots=cand_src,
        target_slots=cand_tgt_slot.unsqueeze(-1).expand(C, L),
        ships=cand_send,
        eta=cand_eta,
        valid=cand_active & cand_valid.unsqueeze(-1),
        player_id=pid,
    )
    score = score_candidates(
        garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), launches=launches, player_id=pid,
    )                                                                            # [C]
    score = torch.where(cand_valid, score, torch.full_like(score, float("-inf")))
    if bool(getattr(config, "enable_ppo_rerank", False)) and C > 0:
        ppo_feat = _ppo_candidate_features(
            score=score, obs=obs, cand_src=cand_src, cand_send=cand_send, cand_eta=cand_eta,
            cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_is_def=cand_is_def,
            cand_valid=cand_valid,
        )
        score = torch.where(
            cand_valid,
            score + _ppo_candidate_bonus(ppo_feat, cand_valid, weight=float(config.ppo_bonus_weight), center=True),
            score,
        )
    wave_entries, leftover = _greedy_select(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=obs.ships.to(dtype).clone(),
        target_exists=target_exists, roi_threshold=float(config.roi_threshold),
        ship_spend_penalty=float(config.greedy_ship_spend_penalty),
        defense_spend_discount=float(config.greedy_defense_spend_discount),
        lookahead_weight=(
            float(config.greedy_lookahead_weight)
            if bool(config.enable_greedy_lookahead)
            else 0.0
        ),
    )
    if not bool(config.enable_regroup):
        return wave_entries
    enemy_mass = cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)  # [P]
    if bool(config.enable_potential_risk):
        # Precautionary potential-attack threat (enemy planets, sun-LOS-gated,
        # discounted by friendly support) biases the regroup to pre-strengthen the
        # most exposed owned planets. Additive — risk_blend_weight=0 == original.
        enemy_mass = enemy_mass + float(config.risk_blend_weight) * potential_attack_risk(
            obs, cache, horizon=float(K_eta), player_id=pid, config=config,
        )
    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=config, H=H,
    )
    return concat_launch_entries([wave_entries, regroup_entries])
def run_turn(obs_tensors: dict, *, config: ProducerLiteConfig, player_count: int, memory) -> dict:
    """Full per-turn pipeline: build movement → plan single-size waves + regroup → emit.
    ``memory`` must expose a mutable ``movement`` attribute (the rolling cache).
    """
    device = obs_tensors["planets"].device
    obs = parse_obs(obs_tensors)
    P = obs.P
    if P == 0:
        return empty_action_row(device)
    movement = ensure_planet_movement(
        obs_tensors=obs_tensors,
        expected_cfg=_movement_config(config, player_count=int(player_count)),
        cached_movement=getattr(memory, "movement", None),
    )
    memory.movement = movement
    cache = build_distance_cache(movement, max_k=int(config.horizon))
    H = int(config.horizon)
    status = movement.garrison_status(max_horizon=H)
    alive_by_step = movement.alive_by_step[: H + 1]
    entries = plan_lite_waves(
        movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
        garrison_status=status, prod=movement.planet_prod,
        alive_by_step=alive_by_step, config=config, player_count=int(player_count),
    )
    entries = disambiguate_duplicate_launches(entries)
    launches = infer_planned_launches_from_entries(
        obs_tensors=obs_tensors, movement=movement, entries=entries, player_id=int(obs.player_id),
    )
    apply_private_planned_launches(
        movement=movement, launches=launches, owner_id=int(obs.player_id),
        obs_tensors=obs_tensors,
    )
    planet_ids = obs_tensors["planets"][..., 0].long()
    return entries_to_sparse_payload(entries, planet_ids=planet_ids)
# 4P FFA preset — only the knobs that differ from the 2P default. 
CONFIG_4P = dataclasses.replace(
    ProducerLiteConfig(),
    horizon=13,
    max_sources_per_lane=6,
    max_defensive_targets=4,
    max_regroup_time=6.0,
    max_regroup_targets_per_source=8,
    enable_single_source_size_variants=False,
    single_source_size_fracs=(1.0,),
    greedy_ship_spend_penalty=0.0,
    enable_greedy_lookahead=True,
    greedy_lookahead_weight=0.25,
    risk_blend_weight=0.5,                   # damp the precaution in diffuse 4p FFA
    ffa_neutral_prod_weight=0.0,
    ffa_enemy_prod_weight=0.0,
    ffa_leader_focus_bonus=0.0,
    ffa_leader_prod_gap=0.0,
    ffa_weak_enemy_penalty=0.0,
    max_strike_sources=3,                    # leaner combined strikes in 4p FFA
    enable_ppo_rerank=True,
    ppo_bonus_weight=1.0,
)
def _config_for(player_count: int) -> ProducerLiteConfig:
    return CONFIG_4P if int(player_count) >= 4 else ProducerLiteConfig()
class ProducerLiteMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None
    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None
class ProducerLiteRuntime:
    def __init__(self, memory: ProducerLiteMemory | None = None) -> None:
        self.memory = memory if memory is not None else ProducerLiteMemory()
    def reset(self) -> None:
        self.memory.reset()
    def tensor_action(self, obs_tensors: dict):
        mem = self.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        config = _config_for(mem.cached_player_count)
        row = run_turn(
            obs_tensors, config=config,
            player_count=int(mem.cached_player_count), memory=mem,
        )
        mem.last_sparse_action_row = row
        return row
_RUNTIME = ProducerLiteRuntime()
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def agent(obs):
    """Single-observation entry point for local play and Kaggle."""
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    player_id = int(player)
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
    with torch.no_grad():
        sparse_row = _RUNTIME.tensor_action(obs_tensors)
    return sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)
