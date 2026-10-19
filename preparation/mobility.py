"""Reproducible pedestrian movement in the existing Cartesian scene geometry.

Only geometry is simulated here: propagation and CQI remain the responsibility
of the supplied WirelessAgent model. Buildings are impassable footprints; the
map rectangle does not imply roads, walkable land or inter-user separation.
"""

from __future__ import annotations

import hashlib
import math
import random

from shapely.geometry import LineString, Point


def _rng(seed: int, user_id: str, purpose: str) -> random.Random:
    """Avoid Python's process-randomized hash and any global RNG mutation."""
    material = f"{seed}\0{user_id}\0{purpose}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))


def _schedule(profile: str, steps: int, interval_s: float, rng):
    """Return movement windows; frame t describes the window ending at t."""
    if profile != "intermittent":
        state = "static" if profile == "static" else "moving"
        return [{"state": state, "start_time_s": 0,
                 "end_time_s": steps * interval_s}]
    schedule, cursor, moving = [], 0, True
    while cursor < steps:
        low_s, high_s = (40, 80) if moving else (20, 40)
        low = max(1, math.ceil(low_s / interval_s))
        high = max(low, math.floor(high_s / interval_s))
        length = rng.randint(low, high)
        # Even short simulations include a pause when >=2 intervals exist.
        if cursor == 0 and steps > 1:
            length = min(length, steps - 1)
        end = min(steps, cursor + length)
        schedule.append({"state": "moving" if moving else "paused",
                         "start_time_s": cursor * interval_s,
                         "end_time_s": end * interval_s})
        cursor, moving = end, not moving
    return schedule


def _advance(position, heading, distance, turn_sign, polygons, bounds):
    """Test quantized endpoints and their complete straight movement segments."""
    x, y, z = position
    min_x, max_x, min_y, max_y = bounds
    offsets = [0.0]
    for degrees in range(15, 180, 15):
        angle = math.radians(degrees) * turn_sign
        offsets.extend((angle, -angle))
    offsets.append(math.pi)
    for offset in offsets:
        candidate_heading = heading + offset
        next_x = round(x + distance * math.cos(candidate_heading), 6)
        next_y = round(y + distance * math.sin(candidate_heading), 6)
        if not (min_x <= next_x <= max_x and min_y <= next_y <= max_y):
            continue
        if (next_x, next_y) == (x, y):
            continue
        segment = LineString(((x, y), (next_x, next_y)))
        if any(polygon.intersects(segment) for polygon in polygons):
            continue
        return (next_x, next_y, z), candidate_heading, "moving"
    return position, heading, "blocked"


def simulate_users(initial_rows, buildings, bounds, *, duration_s=300,
                   interval_s=10, seed=20261013, moving_fraction=2 / 3,
                   intermittent_fraction=0.5, speed_min_mps=0.4,
                   speed_max_mps=0.8):
    """Return ``(frames, profiles)`` with an unchanged frame-zero position.

    ``intermittent_fraction`` is the fraction of *moving* users that pause.
    Fractions are converted to user counts by rounding half up. Assignment
    depends only on the seed and IDs, never on requests, locations or CQI.
    Frame zero and both duration endpoints are included (31 default frames).
    Positions have six decimal places; step distances use those positions.
    Consequently actual speed may differ from nominal by <1 micrometre/s
    for the default 10 s interval. Movement states summarize the preceding
    interval; ``initial`` has zero distance and speed. Profiles contain
    half-open movement/pause windows, with endpoints in simulation seconds.
    """
    parameters = (duration_s, interval_s, moving_fraction, intermittent_fraction,
                  speed_min_mps, speed_max_mps)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in parameters):
        raise ValueError("Movement parameters must be finite numbers")
    if duration_s <= 0 or interval_s <= 0 or interval_s > duration_s:
        raise ValueError("Require 0 < interval_s <= duration_s")
    step_count = duration_s / interval_s
    if not math.isclose(step_count, round(step_count), rel_tol=0, abs_tol=1e-9):
        raise ValueError("duration_s must be an exact multiple of interval_s")
    if not (0 <= moving_fraction <= 1 and 0 <= intermittent_fraction <= 1):
        raise ValueError("Mobility fractions must lie in [0, 1]")
    if not (0 < speed_min_mps <= speed_max_mps):
        raise ValueError("Require 0 < speed_min_mps <= speed_max_mps")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if len(bounds) != 4 or not all(math.isfinite(value) for value in bounds):
        raise ValueError("bounds must contain four finite coordinates")
    min_x, max_x, min_y, max_y = bounds
    if min_x >= max_x or min_y >= max_y:
        raise ValueError("Bounds must be ordered min_x, max_x, min_y, max_y")
    rows = list(initial_rows)
    if not rows:
        raise ValueError("At least one initial user is required")
    user_ids = [str(row["RX_ID"]) for row in rows]
    if any(not user_id.strip() for user_id in user_ids) or len(set(user_ids)) != len(rows):
        raise ValueError("RX_ID must be nonempty and unique within a scene")
    polygons = [building["polygon"] for building in buildings]
    positions = [tuple(float(row[key]) for key in ("X", "Y", "Z")) for row in rows]
    for user_id, position in zip(user_ids, positions):
        x, y, z = position
        if (not all(math.isfinite(value) for value in position)
                or not (min_x <= x <= max_x and min_y <= y <= max_y)
                or any(polygon.intersects(Point(x, y)) for polygon in polygons)):
            raise ValueError(f"Invalid initial position for RX_ID={user_id}")
        if any(value != round(value, 6) for value in position):
            raise ValueError("Initial positions must already have <=6 decimal places")

    count = len(rows)
    moving_count = math.floor(count * moving_fraction + 0.5)
    intermittent_count = math.floor(moving_count * intermittent_fraction + 0.5)
    ranking = sorted(user_ids, key=lambda uid: (_rng(seed, uid, "assignment").random(), uid))
    intermittent_ids = set(ranking[:intermittent_count])
    moving_ids = set(ranking[:moving_count])
    profiles, rngs, headings = [], [], []
    steps = round(step_count)
    for user_id in user_ids:
        profile = ("intermittent" if user_id in intermittent_ids else
                   "continuous" if user_id in moving_ids else "static")
        speed = (_rng(seed, user_id, "speed").uniform(speed_min_mps, speed_max_mps)
                 if user_id in moving_ids else 0.0)
        schedule = _schedule(profile, steps, interval_s, _rng(seed, user_id, "schedule"))
        profiles.append({"user_id": user_id, "mobility_profile": profile,
                         "nominal_speed_mps": speed, "schedule": schedule,
                         "pause_intervals_s": [[window["start_time_s"], window["end_time_s"]]
                                               for window in schedule if window["state"] == "paused"]})
        rng = _rng(seed, user_id, "direction")
        headings.append(rng.uniform(-math.pi, math.pi))
        rngs.append(rng)

    frames, cumulative = [], [0.0] * count
    for frame_index in range(steps + 1):
        frame = []
        for index, (user_id, profile) in enumerate(zip(user_ids, profiles)):
            previous = positions[index]
            state = "initial"
            if frame_index:
                start_time = (frame_index - 1) * interval_s
                state = next(window["state"] for window in profile["schedule"]
                             if window["start_time_s"] <= start_time < window["end_time_s"])
                if state == "moving":
                    rng = rngs[index]
                    heading = headings[index] + rng.gauss(0, math.radians(9))
                    positions[index], headings[index], state = _advance(
                        previous, heading, profile["nominal_speed_mps"] * interval_s,
                        rng.choice((-1, 1)), polygons, bounds)
            step_distance = math.dist(previous, positions[index])
            cumulative[index] += step_distance
            frame.append({"RX_ID": user_id,
                          **dict(zip(("X", "Y", "Z"), positions[index])),
                          "Mobility_Profile": profile["mobility_profile"],
                          "Movement_State": state, "Step_Distance_m": step_distance,
                          "Speed_mps": step_distance / interval_s,
                          "Cumulative_Distance_m": cumulative[index]})
        frames.append(frame)
    return frames, profiles
