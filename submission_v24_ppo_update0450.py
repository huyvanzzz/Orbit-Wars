from __future__ import annotations
# Generated from submission_v24_good.py + PPO checkpoint ppo_policy_update_0450.npz.
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
# PPO checkpoint update 0450 reranker (embedded weights)
# ===========================================================================
_PPO_W1_DATA = [[-0.10864182561635971, -0.0729822963476181, -0.07487864792346954, -0.11675102263689041, 0.014957687817513943, 0.036315083503723145,
  0.05853640288114548, -0.11363735049962997, 0.17593711614608765, -0.18113721907138824, -0.07702422142028809, 0.007663970347493887,
  0.06066569313406944, 0.12887884676456451, 0.1964855194091797, -0.1831909567117691, 0.06075681746006012, 0.0326828695833683,
  0.18359306454658508, 0.1621597409248352, 0.009373164735734463, -0.020451733842492104, -0.043885137885808945, 0.10905832052230835,
  0.3942461907863617, 0.04316039755940437, 0.0626215785741806, -0.0931052565574646, 0.01125357486307621, 0.04657575115561485,
  0.3513936996459961, 0.0847179964184761, 0.19376830756664276, -0.023920871317386627, 0.17801514267921448, 0.03776482865214348,
  -0.0019642403349280357, -0.11049927771091461, -0.15789103507995605, 0.1266786754131317, 0.06322189420461655, -0.16263385117053986,
  0.06083822250366211, 0.25230905413627625, 0.19394856691360474, 0.04233301803469658, 0.05824873596429825, -0.17067073285579681,
  0.10987397283315659, 0.018233347684144974, -0.25933703780174255, 0.3945412039756775, -0.050155315548181534, 0.1927170604467392,
  0.28103107213974, 0.10582038015127182, 0.13612748682498932, -0.07752162963151932, 0.1952858567237854, -0.21063381433486938,
  0.37573111057281494, 0.04120338708162308, -0.08850730955600739, 0.006830842234194279, -0.3075896203517914, -0.09849080443382263,
  -0.10274355858564377, 0.1974726915359497, -0.05638609826564789, -0.26637062430381775, -0.08122710883617401, -0.1784379780292511,
  0.058490902185440063, 0.05428937450051308, 0.09211667627096176, -0.10308364033699036, -0.17293915152549744, 0.05564340576529503,
  -0.04624251276254654, -0.05814402550458908, 0.017490707337856293, -0.017941763624548912, 0.07779940962791443, -0.2544226348400116,
  0.05704561248421669, 0.02771281637251377, -0.03409009799361229, -0.0056450520642101765, 0.0951663926243782, -0.08904711157083511,
  0.0028703731950372458, -0.08196602016687393, -0.058126382529735565, 0.0496646873652935, 0.2896437644958496, -0.06085231155157089],
 [-0.21558938920497894, 0.059651512652635574, -0.20020253956317902, 0.10635440796613693, -0.05854953080415726, 0.09727278351783752,
  0.02437768690288067, 0.0511477068066597, -0.1409992128610611, 0.10120132565498352, 0.0487104095518589, -0.06406006217002869,
  -0.002397261094301939, -0.13994601368904114, -0.13772322237491608, 0.18957914412021637, 0.13393552601337433, 0.1411045342683792,
  -0.10661759227514267, -0.115507110953331, 0.011657802388072014, 0.1465914398431778, -0.030180251225829124, -0.07271758466959,
  0.2883858382701874, 0.14811396598815918, -0.12007090449333191, 0.11823344230651855, 0.04400596395134926, 0.050691913813352585,
  -0.0015033369418233633, -0.07665221393108368, -0.20832505822181702, -0.15506698191165924, -0.1319332867860794, 0.06782743334770203,
  0.0858566090464592, 0.10973779857158661, 0.10245951265096664, 0.0049645318649709225, -0.034330204129219055, -0.2723774015903473,
  0.01220629457384348, -0.1297842264175415, 0.3738240599632263, 0.11044137924909592, -0.07553046941757202, 0.13096822798252106,
  -0.12279310822486877, -0.020283501595258713, -0.14189395308494568, -0.10348191112279892, 0.024929245933890343, -0.11693625897169113,
  -0.1171245202422142, -0.10639353096485138, -0.07715439051389694, 0.033085521310567856, -0.17348238825798035, 0.1202317625284195,
  0.2909577786922455, 0.12982133030891418, 0.07960367947816849, -0.03883225843310356, 0.21128281950950623, 0.05992121249437332,
  0.09721000492572784, -0.11971220374107361, -0.13204796612262726, 0.1351345330476761, 0.09466254711151123, 0.16352854669094086,
  -0.020491112023591995, 0.05960850790143013, -0.043004151433706284, 0.06750413775444031, 0.12613220512866974, -0.029799962416291237,
  -0.19855418801307678, 0.035052698105573654, 0.12359235435724258, -0.0682574063539505, -0.03850041329860687, 0.04641245678067207,
  -0.05279485136270523, -0.15402185916900635, 0.0422307550907135, 0.29726287722587585, -0.025576511397957802, 0.03548276051878929,
  0.1575690656900406, 0.07220776379108429, 0.016377730295062065, 0.017807044088840485, -0.048761457204818726, 0.03425788879394531],
 [1.2214607000350952, 0.02824946865439415, 0.04528418928384781, 0.027083737775683403, 0.007082303520292044, -0.036267027258872986,
  -0.052048537880182266, -1.1144095659255981, -0.04392195865511894, -0.004216655623167753, 0.12418897449970245, -0.02184385620057583,
  -0.034873634576797485, -0.08153684437274933, -0.016362685710191727, 0.018529843538999557, -0.0389925092458725, -0.026190243661403656,
  0.09166179597377777, 0.09677796065807343, 0.02054239995777607, 0.006214152090251446, 0.028379661962389946, -0.04040491580963135,
  0.8147825598716736, -0.025043169036507607, -0.0255842674523592, 0.00266835605725646, 0.025698579847812653, -0.04374251887202263,
  0.08937741816043854, -0.0063347406685352325, -0.01008800882846117, 0.012737050652503967, -0.02287309244275093, -0.03632659092545509,
  -0.007620489224791527, 0.045647405087947845, 0.0822051391005516, 0.017403297126293182, -0.10936075448989868, 1.7169076204299927,
  -0.04272991791367531, 0.032039888203144073, -1.6629897356033325, -0.031146543100476265, -0.02166138030588627, -0.012705033645033836,
  -0.09744322299957275, -0.018913017585873604, -0.6689820885658264, 0.21873529255390167, 0.010736512020230293, 0.0752287209033966,
  0.12276294082403183, 0.07808634638786316, 0.05858570337295532, -0.03305589407682419, 0.0033859852701425552, -0.06692378222942352,
  0.7789009213447571, -0.0324479304254055, 0.033582653850317, -0.012948834337294102, 0.05875108018517494, 0.11912129074335098,
  0.017736196517944336, 0.03205374628305435, 0.05504406616091728, -0.12263511121273041, 0.036716897040605545, -0.014830626547336578,
  -0.053833793848752975, -0.09458526223897934, 0.027636807411909103, 0.06490656733512878, 0.015364321880042553, -0.116063691675663,
  1.0220953226089478, 0.04326195269823074, -0.013777117244899273, 0.01601971685886383, 0.05411124229431152, -0.09691832959651947,
  -0.04209502041339874, -0.015749232843518257, 0.051623232662677765, -0.002967263339087367, 0.07134939730167389, 0.040091246366500854,
  -0.00670580705627799, 0.035345714539289474, 0.0358084961771965, -0.03308567777276039, 0.11154580116271973, 0.04042159020900726],
 [0.10457219183444977, 0.2580980360507965, -0.6159651279449463, -0.06308281421661377, 0.10780300945043564, 0.30564552545547485,
  0.20925664901733398, 0.145853653550148, 0.080523781478405, 0.19515949487686157, -0.12214436382055283, -0.045389171689748764,
  0.10072718560695648, 0.10766306519508362, 0.04621932655572891, -0.07594810426235199, 0.4572390019893646, 0.3513040542602539,
  -0.010596868582069874, -0.00917371641844511, 0.08464935421943665, 0.6203198432922363, -0.1487308144569397, -0.36765816807746887,
  -0.7575666904449463, 0.5673336386680603, -0.5464543700218201, -0.06001419946551323, 0.0754287913441658, 0.19008907675743103,
  -0.09737321734428406, 0.06792187690734863, 0.08389820158481598, -0.6751836538314819, 0.07643474638462067, 0.2093852013349533,
  0.25949743390083313, 0.6727879643440247, -0.10387487709522247, -0.20941807329654694, 0.14821401238441467, 0.24355581402778625,
  0.18199053406715393, 0.0288567915558815, -0.2380405068397522, 0.33067718148231506, -0.261241614818573, -0.03584213927388191,
  0.09955374151468277, -0.1176222413778305, 0.3832171559333801, -0.488145649433136, -0.09461480379104614, -0.0008038546075113118,
  -0.012684157118201256, 0.013382521457970142, 0.008976136334240437, 0.0050719184800982475, 0.05825367569923401, -0.009684045799076557,
  -0.7110952138900757, 0.3707204759120941, -0.07160595059394836, -0.11927243322134018, -0.12255048751831055, -0.13890428841114044,
  -0.07426127046346664, 0.013340530917048454, -0.2650218605995178, 0.014196714386343956, 0.4836622476577759, -0.03111357055604458,
  0.1753539741039276, 0.1803780198097229, -0.09432850033044815, -0.11586945503950119, -0.05289760231971741, 0.09312497079372406,
  -0.014999505132436752, -0.10305295884609222, 0.4085234999656677, -0.31484273076057434, 0.0268825963139534, 0.009357351809740067,
  0.1476137489080429, -0.5890504717826843, 0.05152195692062378, 0.48063400387763977, 0.01267827209085226, -0.11159056425094604,
  0.6452159285545349, -0.1022181287407875, -0.12458079308271408, 0.13500367105007172, -0.014843033626675606, -0.13400761783123016],
 [0.06961173564195633, 0.3972247540950775, -0.8331820964813232, -0.36333927512168884, 0.40584155917167664, 0.5749356150627136,
  0.5366528630256653, -0.03623262792825699, 0.3621256649494171, 0.008974514901638031, -0.5366050004959106, -0.31892791390419006,
  0.45833727717399597, 0.5280084013938904, 0.15177856385707855, -0.3105481266975403, 0.676021158695221, 0.5700979828834534,
  -0.004022620152682066, 0.13933400809764862, 0.3695595860481262, 0.5223430395126343, -0.4703516662120819, -0.44429638981819153,
  -0.1500782072544098, 0.6322179436683655, -0.4851723611354828, -0.3776693642139435, 0.3207976520061493, 0.5193781852722168,
  -0.028378654271364212, 0.3986726999282837, 0.3866468667984009, -0.627147912979126, 0.3645356595516205, 0.5263168811798096,
  0.4597773253917694, 0.4572831690311432, -0.47334805130958557, 0.02367265895009041, 0.5151801109313965, 0.12461455911397934,
  0.522242546081543, -0.06863769143819809, -0.130111962556839, 0.5848572850227356, -0.4052734971046448, -0.185317724943161,
  0.4720730185508728, -0.38007742166519165, -0.03626597300171852, -0.10240861773490906, -0.42177289724349976, 0.0010265229502692819,
  -0.08317158371210098, 0.23695988953113556, 0.18312996625900269, -0.26512759923934937, 0.2258574217557907, -0.0022072051651775837,
  -0.15937687456607819, 0.5697237849235535, -0.4329543113708496, -0.40027502179145813, -0.39223554730415344, -0.5583893656730652,
  -0.4491760730743408, 0.05153926461935043, -0.6063500642776489, 0.06352454423904419, 0.4721404016017914, -0.12259206175804138,
  0.5409523248672485, 0.5389664173126221, 0.12535719573497772, -0.49034011363983154, -0.19784210622310638, 0.47388190031051636,
  0.01186321396380663, -0.4593797028064728, 0.5671307444572449, -0.5368834137916565, 0.2778978943824768, 0.08256110548973083,
  0.4959888458251953, -0.5156716704368591, 0.28068679571151733, 0.5822134613990784, 0.239003986120224, -0.4806610643863678,
  0.582032322883606, -0.47528785467147827, -0.4755416214466095, 0.4789733588695526, -0.10991528630256653, -0.4985155463218689],
 [0.2745422124862671, 0.13697464764118195, -0.13360728323459625, -0.14142191410064697, 0.15784063935279846, 0.07664166390895844,
  0.09027601778507233, -0.350243479013443, 0.03725820779800415, -0.3183082640171051, -0.054126087576150894, -0.2673013508319855,
  0.10664910823106766, 0.020329294726252556, 0.04538164660334587, -0.0829213336110115, 0.11898091435432434, 0.12711404263973236,
  0.09488190710544586, -0.002193516120314598, 0.1457187533378601, 0.12006332725286484, -0.16916312277317047, -0.11622612923383713,
  -0.024991273880004883, 0.1044236421585083, -0.11497672647237778, -0.15721440315246582, 0.1808304786682129, 0.12093599140644073,
  -0.04481736570596695, 0.0906265527009964, 0.00674300966784358, -0.11778228729963303, 0.031212128698825836, 0.09755412489175797,
  0.12654848396778107, 0.11113118380308151, -0.05289248749613762, 0.5588691234588623, 0.013871429488062859, 0.26268190145492554,
  0.09523199498653412, 0.04931739717721939, -0.2454901784658432, 0.10416535288095474, -0.15276339650154114, -0.033078182488679886,
  0.0725962296128273, -0.16539184749126434, -0.2619735598564148, -0.5254075527191162, -0.09995249658823013, 0.048414312303066254,
  -0.050049807876348495, 0.13552530109882355, 0.15376058220863342, -0.20125967264175415, -0.005266700871288776, -0.052958227694034576,
  0.07929236441850662, 0.09565897285938263, -0.09264571219682693, -0.1441696435213089, 0.06519336998462677, 0.014987192116677761,
  -0.03991701453924179, 0.08603431284427643, -0.09311222285032272, 0.027930188924074173, 0.12484285235404968, -0.05347544327378273,
  0.06712815910577774, 0.05733334273099899, 0.35222360491752625, -0.0420212596654892, -0.06309697031974792, 0.12234007567167282,
  0.32440125942230225, -0.08379332721233368, 0.1616390347480774, -0.11669962108135223, 0.14602148532867432, -0.07448563724756241,
  0.0750097781419754, -0.15338341891765594, 0.2961357831954956, 0.2834061086177826, 0.10723868757486343, -0.043661754578351974,
  0.09702453762292862, -0.06387143582105637, -0.08433908224105835, 0.10855042189359665, 0.026485208421945572, -0.05233774706721306],
 [0.17034251987934113, 0.01277379784733057, -0.02489609271287918, -0.08328613638877869, 0.043746981769800186, -0.04609502851963043,
  -0.05769188329577446, -0.22657321393489838, -0.08574797213077545, -0.26846063137054443, 0.015280371531844139, -0.14375965297222137,
  0.0703573077917099, -0.07255388051271439, -0.09171155095100403, 0.06491890549659729, -0.03035518154501915, 0.006254997570067644,
  -0.03760527819395065, -0.10672663897275925, 0.026319507509469986, 0.004461097531020641, -0.05874003469944, -0.031329091638326645,
  -0.1761326938867569, 0.002723930636420846, -0.011268103495240211, -0.08252613246440887, 0.14729799330234528, 0.03969348222017288,
  -0.20674727857112885, 0.04148048907518387, -0.11730042845010757, 0.0476800873875618, -0.10996458679437637, -0.010848392732441425,
  0.008884305134415627, -0.04682449996471405, 0.05376414582133293, 0.48787233233451843, -0.08045769482851028, 0.1647171825170517,
  0.0021613813005387783, -0.13425736129283905, -0.15232372283935547, 0.03284443914890289, -0.043291110545396805, 0.051490720361471176,
  -0.004918541293591261, -0.046651486307382584, -0.153468057513237, -0.6964833736419678, -0.0509776771068573, -0.05382132530212402,
  -0.19836729764938354, 0.035346731543540955, 0.014942159876227379, -0.11187104135751724, -0.07691612094640732, 0.09911829233169556,
  -0.1364159882068634, -0.013435281813144684, 0.007712459657341242, -0.05140648037195206, 0.16401773691177368, 0.085247702896595,
  0.028075920417904854, -0.007613813038915396, 0.004513583146035671, 0.235568106174469, -0.050842493772506714, 0.056311652064323425,
  -0.05230208486318588, -0.00938450638204813, 0.22505605220794678, 0.03493008390069008, 0.03998752310872078, 0.00895305909216404,
  0.15939629077911377, -0.0034183866810053587, 0.044286634773015976, -0.0037856821436434984, 0.0727553442120552, 0.07681463658809662,
  -0.041756805032491684, -0.05018414556980133, 0.18701981008052826, 0.16896727681159973, 0.06999199092388153, 0.0365762822329998,
  -0.007087439764291048, 0.022450655698776245, 0.006910751573741436, -0.008968577720224857, -0.14819517731666565, 0.002961177146062255],
 [-0.21346062421798706, -0.056187719106674194, 0.014280285686254501, 0.04991880804300308, 0.0010777461575344205, -0.0436834841966629,
  -0.0613592192530632, 0.24131514132022858, -0.08670251071453094, -0.043255437165498734, 0.04164958372712135, -0.004053027369081974,
  0.06393878906965256, -0.017863091081380844, -0.21804223954677582, 0.0817292258143425, -0.007688883692026138, 0.018315890803933144,
  -0.24417774379253387, -0.21734316647052765, -0.01387973502278328, -0.0253454577177763, -0.06338726729154587, 0.050104133784770966,
  -0.4939846992492676, 0.028416382148861885, 0.08136556297540665, -0.0289533082395792, 0.16845354437828064, -0.004643615335226059,
  -0.5408818125724792, 0.011798739433288574, -0.046334706246852875, 0.04530688747763634, -0.12259291112422943, -0.07158764451742172,
  0.001510265632532537, -0.0993637964129448, 0.05231589078903198, 0.145257830619812, -0.08302652835845947, -0.3584508001804352,
  -0.02136651612818241, -0.42393019795417786, 0.3325202167034149, 0.002191311912611127, 0.03677886351943016, 0.181076318025589,
  -0.0599876306951046, 0.014634799212217331, 0.3733334243297577, -0.6823989152908325, -0.03382503241300583, -0.24230270087718964,
  -0.47180792689323425, -0.019292408600449562, -0.11447196453809738, 0.010311642661690712, -0.1316799819469452, 0.300387978553772,
  -0.479128897190094, -0.021624870598316193, 0.05280618369579315, 0.026571068912744522, 0.133985236287117, 0.10486788302659988,
  0.022652139887213707, -0.2960532009601593, 0.04055710509419441, 0.47372326254844666, -0.025005927309393883, 0.22091801464557648,
  -0.0695541575551033, -0.04383580759167671, 0.006086093373596668, -0.002339469501748681, 0.16929058730602264, -0.031220484524965286,
  -0.1390785574913025, 0.01937098614871502, 0.005114315077662468, 0.016110630705952644, 0.017110083252191544, 0.3711986839771271,
  -0.01803096942603588, -0.001322138705290854, -0.012012283317744732, 0.1242508590221405, -0.048733774572610855, -0.00027129691443406045,
  -0.033680617809295654, -0.02858191914856434, 0.013817298226058483, -0.020706696435809135, -0.4133910536766052, 0.07702908664941788],
 [-0.03248726949095726, -0.04141709953546524, 0.03539732098579407, 0.03498116135597229, -0.05373404920101166, -0.031543247401714325,
  0.0021998973097652197, 0.21461671590805054, -0.026258990168571472, 0.072953000664711, 0.009515990503132343, 0.016602544113993645,
  -0.017118729650974274, 0.045903198421001434, -0.01718423143029213, 0.02012842707335949, -0.03715396299958229, 0.0033016942907124758,
  -0.11466912180185318, 0.006500950548797846, -0.06556981056928635, -0.0046502575278282166, -0.013931198976933956, 0.056394461542367935,
  -0.4226510524749756, -0.018159301951527596, 0.022333575412631035, 0.04626704379916191, -0.0104020144790411, -0.020403189584612846,
  -0.20097698271274567, -0.02998119406402111, 0.05665997415781021, 0.01460206974297762, 0.018991751596331596, -0.03165515139698982,
  -0.019209356978535652, -0.05960006266832352, 0.0025116801261901855, 0.06303012371063232, -0.0015751513419672847, -0.12003670632839203,
  0.011265064589679241, -0.03915619105100632, 0.1320406049489975, -0.020672336220741272, 0.02730563096702099, 0.03173212334513664,
  -0.0372990146279335, 0.020210368558764458, 0.33386480808258057, 0.06294956058263779, -0.004830287769436836, -0.09841030836105347,
  -0.037032224237918854, -0.06646784394979477, -0.08681727945804596, 0.052565038204193115, -0.022645603865385056, 0.04218899458646774,
  -0.4571484923362732, -0.005401778034865856, 0.04061736911535263, 0.0536825992166996, -0.009303949773311615, -0.06529074162244797,
  -0.012546969577670097, -0.11642710864543915, -0.027315063402056694, -0.0038739724550396204, -0.026527468115091324, 0.012788131833076477,
  -0.02243548445403576, 0.016164548695087433, -0.06043751537799835, -0.004225144162774086, 0.007758783642202616, -0.05886643007397652,
  -0.02685922384262085, 0.04815707728266716, 0.004787680227309465, 0.02149783819913864, -0.049660343676805496, 0.15238527953624725,
  -0.012987303547561169, -0.0227121040225029, 0.004876127932220697, 0.11273851990699768, -0.08657382428646088, 0.0017134945373982191,
  -0.029020138084888458, 0.007021565921604633, 0.021570883691310883, 0.00493529811501503, -0.12130970507860184, 0.009186557494103909],
 [-0.04128633812069893, 0.05051436275243759, -0.3349854052066803, 0.12001389265060425, 0.051890779286623, 0.16002106666564941,
  0.1589096337556839, -0.020251069217920303, 0.030938778072595596, -0.00983115192502737, -0.031572677195072174, -0.05249140039086342,
  -0.0028549267444759607, -0.027747759595513344, -0.1095670536160469, 0.035681698471307755, 0.23393316566944122, 0.12242070585489273,
  -0.04025321826338768, -0.15094800293445587, -0.023260245099663734, 0.20532174408435822, -0.09461274743080139, -0.06003662943840027,
  0.13476033508777618, 0.17438045144081116, -0.12979617714881897, 0.059982772916555405, -0.010617700405418873, 0.10954444855451584,
  0.051233112812042236, -0.031772296875715256, 0.0029619100969284773, -0.1250843107700348, -0.0320567823946476, 0.09410543739795685,
  0.05422371253371239, 0.16275794804096222, 0.00419014785438776, -0.026445310562849045, 0.15517958998680115, -0.04894595593214035,
  0.13067518174648285, -0.34792864322662354, 0.059932153671979904, 0.1644858419895172, -0.051416102796792984, 0.07774824649095535,
  -0.04167851060628891, -0.054912157356739044, -0.09168782830238342, -0.15785394608974457, 0.00913897342979908, -0.09678489714860916,
  -0.32472875714302063, -0.08208361268043518, -0.012887678109109402, -0.042790647596120834, -0.005896345246583223, 0.22314544022083282,
  0.1372060477733612, 0.17148253321647644, -0.035866089165210724, -0.042000215500593185, 0.015203061513602734, -0.08203265070915222,
  -0.008416573517024517, -0.12639795243740082, -0.23701144754886627, 0.27265819907188416, 0.015342441387474537, 0.10460945963859558,
  0.1348542422056198, 0.2004542201757431, -0.028112316504120827, -0.04555622115731239, 0.13142192363739014, 0.005530559923499823,
  -0.07297158986330032, -0.04647814482450485, 0.20190659165382385, -0.1437842845916748, -0.0641457810997963, 0.12147344648838043,
  0.0889812782406807, -0.13669738173484802, 0.05071325972676277, 0.14758826792240143, -0.06283893436193466, -0.009787829592823982,
  0.20738521218299866, 0.002545235212892294, -0.04691405966877937, 0.03613080456852913, -0.15042416751384735, -0.11413533240556717],
 [-0.09960780292749405, -0.027365440502762794, 0.13872399926185608, 0.01600373163819313, -0.05554548650979996, -0.06926281005144119,
  -0.1124168187379837, 0.09090642631053925, -0.1391693502664566, -0.037123922258615494, 0.10926219820976257, 0.06835424900054932,
  -0.06440318375825882, -0.1329805999994278, -0.08595062047243118, 0.14099007844924927, -0.11496944725513458, -0.050961557775735855,
  -0.05183914303779602, -0.05351852625608444, -0.05187813565135002, -0.03651437163352966, 0.07758807390928268, -0.0029893594328314066,
  -0.021004343405365944, -0.05693588778376579, 0.03867419436573982, 0.06420722603797913, 0.011774668470025063, -0.08329737931489944,
  -0.14163319766521454, -0.03656258061528206, -0.14239643514156342, 0.027900010347366333, -0.14080557227134705, -0.08861281722784042,
  -0.04655614122748375, -0.020712077617645264, 0.09705159068107605, -0.02497093379497528, -0.12659183144569397, -0.19411031901836395,
  -0.13132506608963013, -0.00011767586693167686, 0.19939421117305756, -0.0918973907828331, 0.03318405523896217, 0.07057604193687439,
  -0.10633296519517899, 0.05263805389404297, -0.023876139894127846, -0.38600635528564453, 0.06951043754816055, -0.05477803200483322,
  -0.04849691316485405, 0.008665533736348152, -0.011347820982336998, 0.02967490814626217, -0.16569173336029053, 0.00600814213976264,
  -0.030286021530628204, -0.06436842679977417, 0.09377370029687881, 0.031611792743206024, 0.24563007056713104, 0.16676178574562073,
  0.1030484288930893, -0.02283606119453907, 0.11928240954875946, 0.07387746870517731, -0.020816830918192863, 0.07427400350570679,
  -0.12626823782920837, -0.09145403653383255, -0.03158663585782051, 0.11870313435792923, 0.038854751735925674, -0.03582271933555603,
  -0.06259384751319885, 0.07507994771003723, -0.07960223406553268, 0.06632880121469498, -0.005947123747318983, 0.030513864010572433,
  -0.08852831274271011, 0.044824350625276566, -0.0337502546608448, -0.12746240198612213, -0.0038976159412413836, 0.08177850395441055,
  -0.04593436047434807, 0.08657755702733994, 0.08333984017372131, -0.052939556539058685, -0.03619613125920296, 0.0968063622713089],
 [0.30329644680023193, -0.12834541499614716, 0.1371530443429947, 0.34585079550743103, -0.22030064463615417, -0.150651216506958,
  -0.1622375249862671, -0.26139184832572937, -0.15456172823905945, -0.6241022348403931, 0.18267612159252167, 0.003219298552721739,
  -0.2546563446521759, -0.15387871861457825, -0.3019377589225769, 0.19941659271717072, -0.11787977069616318, -0.1866278350353241,
  -0.42170989513397217, -0.45341646671295166, -0.18568076193332672, -0.20297090709209442, 0.24547959864139557, 0.16600733995437622,
  0.18848204612731934, -0.17773284018039703, 0.14939439296722412, 0.2775685787200928, -0.3486853837966919, -0.2032863348722458,
  0.10085922479629517, -0.2508295774459839, -0.16075032949447632, 0.16462236642837524, -0.14606693387031555, -0.15354260802268982,
  -0.21618923544883728, -0.16412700712680817, 0.201121985912323, 0.5628896951675415, -0.13706417381763458, 0.23501688241958618,
  -0.143222376704216, -0.5751656293869019, -0.32495182752609253, -0.15668714046478271, 0.17074422538280487, 0.3375695049762726,
  -0.20259302854537964, 0.11871135979890823, -0.1795581430196762, -0.2726113200187683, 0.21742989122867584, -0.453578919172287,
  -0.574721097946167, -0.47393932938575745, -0.40425512194633484, 0.05807557329535484, -0.18517209589481354, 0.5378140807151794,
  0.16850025951862335, -0.14343473315238953, 0.15426884591579437, 0.18924111127853394, 0.12544076144695282, 0.08257708698511124,
  0.13673214614391327, -0.47250106930732727, 0.1291961520910263, 0.5040404796600342, -0.13091818988323212, 0.3475680947303772,
  -0.15137693285942078, -0.1921079456806183, 0.317050576210022, 0.15983714163303375, 0.3703359365463257, -0.2493329495191574,
  0.4214709997177124, 0.1976546198129654, -0.20359469950199127, 0.17355117201805115, -0.37283316254615784, 0.45770177245140076,
  -0.18467886745929718, 0.2006916105747223, 0.16125454008579254, -0.35222700238227844, -0.42194077372550964, 0.18617962300777435,
  -0.20019745826721191, 0.20357660949230194, 0.18180644512176514, -0.18414843082427979, -0.46298542618751526, 0.17100465297698975]]
_PPO_B1_DATA = [-0.3210281729698181, -0.007789967115968466, -0.11111212521791458, 0.5919568538665771, -0.1617465317249298, 0.05463016405701637,
 -0.08868328481912613, 0.37238529324531555, -0.44834861159324646, -0.03642842918634415, 0.33469757437705994, 0.061805304139852524,
 -0.2524259686470032, -0.46007850766181946, -0.6327409744262695, 0.579624354839325, 0.07104127109050751, 0.028975799679756165,
 -0.6595418453216553, -0.7128255367279053, -0.18808682262897491, 0.09129252284765244, 0.08551431447267532, 0.012711308896541595,
 -0.379130482673645, 0.08582837879657745, -0.05972606688737869, 0.49534550309181213, -0.27412939071655273, -0.0716572031378746,
 -0.47695523500442505, -0.46042490005493164, -0.5786934494972229, -0.07300840318202972, -0.5198682546615601, -0.03203018382191658,
 0.0262736976146698, 0.04511939734220505, 0.5231447219848633, 0.10777460783720016, -0.2132091373205185, -0.5444853901863098,
 -0.14444157481193542, -0.7864245176315308, 0.45877528190612793, 0.02571331337094307, 0.01086380984634161, 0.609756350517273,
 -0.43273797631263733, 0.0979461744427681, 0.38961631059646606, -0.6652414798736572, 0.23138269782066345, -0.6667841076850891,
 -0.7918047904968262, -0.6186801791191101, -0.6058675050735474, 0.38621842861175537, -0.5316667556762695, 0.7722944617271423,
 -0.38102081418037415, 0.03389214351773262, 0.31142914295196533, 0.11654847860336304, 0.5344558358192444, 0.37419748306274414,
 0.3997785747051239, -0.6620420813560486, -0.00296835508197546, 0.7951529026031494, 0.020248249173164368, 0.6450667977333069,
 -0.12250624597072601, -0.11439771205186844, -0.04291635751724243, 0.3984794318675995, 0.6820745468139648, -0.3038608729839325,
 -0.16086894273757935, 0.19784629344940186, 0.08370203524827957, -0.025013359263539314, -0.5194987058639526, 0.7299827337265015,
 -0.1790873408317566, -0.05304688215255737, 0.04776554927229881, 0.1145344078540802, -0.5790228247642517, 0.3670760989189148,
 0.11896071583032608, 0.3449203073978424, 0.21227802336215973, -0.20853067934513092, -0.7786731719970703, 0.14833950996398926]
_PPO_W2_DATA = [[-0.6066818237304688], [-0.23288391530513763], [0.32632774114608765], [0.24691142141819], [-0.20176264643669128], [-0.2415769398212433],
 [-0.2691647708415985], [0.4305911958217621], [-0.2578682005405426], [0.2948082983493805], [0.25867411494255066], [0.1660062074661255],
 [-0.22191399335861206], [-0.268505722284317], [-0.26665735244750977], [0.2712048292160034], [-0.3166440427303314], [-0.23123875260353088],
 [-0.3271822929382324], [-0.35136476159095764], [-0.1757204383611679], [-0.29324233531951904], [0.22151662409305573], [0.2703300714492798],
 [-0.5389980673789978], [-0.29199546575546265], [0.3083295226097107], [0.2318541258573532], [-0.18016871809959412], [-0.21007686853408813],
 [-0.35121288895606995], [-0.2287036031484604], [-0.2698037326335907], [0.2923477292060852], [-0.26829835772514343], [-0.22151318192481995],
 [-0.21556265652179718], [-0.3731340765953064], [0.26167309284210205], [-0.30852147936820984], [-0.2671642005443573], [-1.0387734174728394],
 [-0.21642348170280457], [-0.44717714190483093], [0.9967256784439087], [-0.27804502844810486], [0.23818905651569366], [0.26723790168762207],
 [-0.23477064073085785], [0.18813228607177734], [0.308302640914917], [-0.7423232793807983], [0.16352242231369019], [-0.3330920934677124],
 [-0.5084823966026306], [-0.2916507422924042], [-0.2707637548446655], [0.17904473841190338], [-0.2548908293247223], [0.4144209027290344],
 [-0.48419278860092163], [-0.23796305060386658], [0.20589222013950348], [0.1936301290988922], [0.38749590516090393], [0.259000301361084],
 [0.23354114592075348], [-0.3107743561267853], [0.22112958133220673], [0.49641722440719604], [-0.2870056927204132], [0.27277955412864685],
 [-0.23014958202838898], [-0.23941513895988464], [-0.19703881442546844], [0.2572658061981201], [0.2696148753166199], [-0.17958180606365204],
 [-0.39706650376319885], [0.17824575304985046], [-0.2725595235824585], [0.23797860741615295], [-0.22099246084690094], [0.38085371255874634],
 [-0.21663370728492737], [0.30441519618034363], [-0.1197950541973114], [-0.326973021030426], [-0.24585527181625366], [0.2138977199792862],
 [-0.288804829120636], [0.2053307294845581], [0.19761888682842255], [-0.18435390293598175], [-0.4441865384578705], [0.22379283607006073]]
_PPO_B2_DATA = [-2.9849326610565186]
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
