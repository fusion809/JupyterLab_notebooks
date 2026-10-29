# COMPLETE SINGLE-CELL WORKFLOW: Mount Stuart descent.
# Fetch the Mount Stuart Road route and QldDem terrain, calculate a downhill
# speed profile, and render a smoothed-camera 3D animation.
#
# Requires: numpy, scipy, requests, matplotlib and ffmpeg.
# Run in a Linux Jupyter environment with an internet connection.

import json
import hashlib
import math
import os
import shutil
import subprocess
import tempfile
import multiprocessing as mp
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import requests
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

import matplotlib as mpl

# Rendering runs in worker processes without a display server. Select a
# non-interactive backend before importing pyplot so workers never initialize
# Qt's Wayland/X11 platform plugins.
mpl.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib import cm


# Keep fetched elevation data and numerical simulation results beside this
# script (or in the notebook's working directory when run as a cell).
try:
    _WORK_DIR = Path(__file__).resolve().parent
except NameError:
    _WORK_DIR = Path.cwd()

DATA_DIR = _WORK_DIR / "mount_stuart_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(*arrays, settings=()):
    digest = hashlib.sha256(repr(tuple(settings)).encode("utf-8"))
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(values.view(np.uint8))
    return digest.hexdigest()[:16]


def _save_npz_atomic(path, **arrays):
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("wb") as cache_file:
        np.savez(cache_file, **arrays)
    temporary_path.replace(path)


# =====================================================================
# 1. Fetch Mount Stuart Road from OpenStreetMap/OSRM
# =====================================================================

print("Fetching Mount Stuart Road route from OpenStreetMap/OSRM...")

def _geocode(address):
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={
            "q": address,
            "format": "json",
            "countrycodes": "au",
            "limit": 10,
        },
        headers={"User-Agent": "Mount-Stuart-terrain-animation/1.0"},
        timeout=30,
    )
    response.raise_for_status()
    results = response.json()

    if not results:
        raise RuntimeError(f"No geocoding result for {address!r}")

    for item in results:
        name = item.get("display_name", "").lower()
        if "stuart drive" in name:
            return float(item["lon"]), float(item["lat"])

    raise RuntimeError(
        f"Could not identify Stuart Drive in geocoding results: {address!r}"
    )


# Approximate Mount Stuart summit lookout and Mount Stuart Road entrance.
# The summit is around 19.3432° S, 146.7810° E.
# The road entrance is around 19.3609° S, 146.8372° E.
summit_lon, summit_lat = 146.7810, -19.3432
base_lon, base_lat = 146.8372, -19.3609

_route_url = (
    "https://router.project-osrm.org/route/v1/driving/"
    f"{summit_lon},{summit_lat};{base_lon},{base_lat}"
)

_route_response = requests.get(
    _route_url,
    params={
        "overview": "full",
        "geometries": "geojson",
        "steps": "true",
    },
    timeout=60,
)
_route_response.raise_for_status()
_route_result = _route_response.json()

if _route_result.get("code") != "Ok":
    raise RuntimeError(f"OSRM routing failed: {_route_result}")

_route = _route_result["routes"][0]
_route_coords = np.asarray(_route["geometry"]["coordinates"], dtype=float)

_route_names = [
    step.get("name", "").strip()
    for leg in _route.get("legs", [])
    for step in leg.get("steps", [])
]

print("OSRM route roads:")
for _name in _route_names:
    if _name:
        print(f"  {_name}")

if not _route_names:
    raise RuntimeError("OSRM returned no named road segments.")

if not any(
    "stuart" in name.lower()
    for name in _route_names
):
    raise RuntimeError(
        "OSRM did not return a route containing a road with 'Stuart' "
        "in its name. Inspect the route before proceeding."
    )


# =====================================================================
# 2. Convert longitude/latitude to local metric coordinates
# =====================================================================

_center_lon = float(np.mean(_route_coords[:, 0]))
_center_lat = float(np.mean(_route_coords[:, 1]))

_m_per_deg_x = 111320.0 * math.cos(math.radians(_center_lat))
_m_per_deg_y = 110574.0


def _local_xy(lon, lat):
    return (
        (np.asarray(lon, dtype=float) - _center_lon) * _m_per_deg_x,
        (np.asarray(lat, dtype=float) - _center_lat) * _m_per_deg_y,
    )


# =====================================================================
# 3. Query the Queensland QldDem elevation service
# =====================================================================

# The identify endpoint matches the identify-style response expected below.
_identify_url = (
    "https://spatial-img.information.qld.gov.au/"
    "arcgis/rest/services/Elevation/QldDem/ImageServer/identify"
)


def _query_qlddem(lons, lats, chunk_size=100, max_retries=3):
    lons = np.asarray(lons, dtype=float).ravel()
    lats = np.asarray(lats, dtype=float).ravel()

    if len(lons) != len(lats):
        raise ValueError(
            f"Longitude and latitude arrays differ in length: "
            f"{len(lons)} != {len(lats)}"
        )

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")

    values = np.full(len(lons), np.nan, dtype=float)
    total = len(lons)

    def _fetch_batch(begin, end):
        geometry = {
            "points": [
                [float(lon), float(lat)]
                for lon, lat in zip(lons[begin:end], lats[begin:end])
            ],
            "spatialReference": {"wkid": 4326},
        }

        params = {
            "geometry": json.dumps(geometry),
            "geometryType": "esriGeometryMultipoint",
            "returnGeometry": "false",
            "returnCatalogItems": "false",
            "f": "json",
        }

        last_error = None

        for attempt in range(max_retries + 1):
            try:
                response = requests.get(
                    _identify_url,
                    params=params,
                    timeout=60,
                )
                response.raise_for_status()
                payload = response.json()

                if "error" in payload:
                    raise RuntimeError(
                        f"QldDem identify error: {payload['error']}"
                    )

                # ArcGIS can return one result directly as an object for a
                # single-point request, or a list for multipoint requests.
                results = payload.get("results")

                if results is None and "value" in payload:
                    results = [payload]

                if not isinstance(results, list):
                    raise RuntimeError(
                        f"Unexpected QldDem identify response: {payload}"
                    )

                if len(results) != end - begin:
                    raise RuntimeError(
                        f"QldDem identify returned {len(results)} results "
                        f"for {end - begin} points"
                    )

                batch_values = np.full(end - begin, np.nan)

                for j, result in enumerate(results):
                    try:
                        batch_values[j] = float(result["value"])
                    except (KeyError, TypeError, ValueError):
                        pass

                if not np.all(np.isfinite(batch_values)):
                    raise RuntimeError(
                        "QldDem identify returned missing or non-finite "
                        "elevations."
                    )

                return batch_values

            except (
                requests.RequestException,
                ValueError,
                RuntimeError,
            ) as exc:
                last_error = exc

                if attempt < max_retries:
                    delay = min(2 ** attempt, 8)
                    print(
                        f"\nQldDem identify failed for samples "
                        f"{begin:,}–{end - 1:,} "
                        f"(attempt {attempt + 1}/{max_retries + 1}): "
                        f"{exc}. Retrying in {delay}s...",
                        flush=True,
                    )
                    time.sleep(delay)

        # Split a persistently failing batch to isolate problematic requests.
        if end - begin > 1:
            midpoint = begin + (end - begin) // 2
            left = _fetch_batch(begin, midpoint)
            right = _fetch_batch(midpoint, end)
            return np.concatenate((left, right))

        raise RuntimeError(
            f"QldDem identify failed for sample {begin}: "
            f"longitude={lons[begin]:.8f}, "
            f"latitude={lats[begin]:.8f}. Last error: {last_error}"
        ) from last_error

    for begin in range(0, total, chunk_size):
        end = min(begin + chunk_size, total)
        values[begin:end] = _fetch_batch(begin, end)

        print(
            f"\rQldDem samples: {end:,}/{total:,}",
            end="",
            flush=True,
        )

    print()

    if not np.all(np.isfinite(values)):
        raise RuntimeError(
            f"QldDem returned "
            f"{np.count_nonzero(~np.isfinite(values))} invalid elevations."
        )

    return values


# =====================================================================
# 4. Sample the route and construct its elevation profile
# =====================================================================

def _haversine(lon1, lat1, lon2, lat2):
    radius = 6371000.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )

    return 2 * radius * math.asin(math.sqrt(a))


_route_s = np.zeros(len(_route_coords), dtype=float)

for _i in range(1, len(_route_coords)):
    _route_s[_i] = (
        _route_s[_i - 1]
        + _haversine(
            *_route_coords[_i - 1],
            *_route_coords[_i],
        )
    )

# Mount Stuart Road is approximately 9 km long.
if not 6000.0 <= _route_s[-1] <= 12000.0:
    raise RuntimeError(
        f"Unexpected Mount Stuart route length: {_route_s[-1]:.1f} m. "
        "Check the OSRM route before querying the terrain."
    )

_profile_s = np.arange(0.0, _route_s[-1], 10.0)

if len(_profile_s) == 0 or _profile_s[-1] < _route_s[-1]:
    _profile_s = np.append(_profile_s, _route_s[-1])

_route_lon = np.interp(
    _profile_s,
    _route_s,
    _route_coords[:, 0],
)
_route_lat = np.interp(
    _profile_s,
    _route_s,
    _route_coords[:, 1],
)

print(f"Route length: {_route_s[-1] / 1000:.3f} km")
_profile_cache_key = _cache_key(
    _route_lon,
    _route_lat,
    settings=("QldDem-route-profile-v1", 10.0),
)
_profile_cache = DATA_DIR / f"route_profile_{_profile_cache_key}.npz"

if _profile_cache.exists():
    with np.load(_profile_cache) as cache:
        _profile_z = cache["elevation"].copy()
    if len(_profile_z) != len(_profile_s):
        raise RuntimeError(f"Invalid route elevation cache: {_profile_cache}")
    print(f"Loaded cached QldDem route profile: {_profile_cache}")
else:
    print("Querying QldDem for the route elevation profile...")
    # Small chunks preserve the conservative request behaviour of the original.
    _profile_z = _query_qlddem(
        _route_lon,
        _route_lat,
        chunk_size=3,
    )

    if len(_profile_z) >= 11:
        _profile_z = savgol_filter(_profile_z, 11, 3)

    _save_npz_atomic(_profile_cache, elevation=_profile_z)

# OSRM was queried summit -> base, so the route is already oriented for descent.
distance = _profile_s.copy()
h = _profile_z.copy()

road_lon = _route_lon.copy()
road_lat = _route_lat.copy()

road_x, road_y = _local_xy(road_lon, road_lat)
road_z = h.copy()


# =====================================================================
# 5. Build the surrounding QldDem terrain mesh
# =====================================================================

_grid_spacing = 65.0
_margin = 1800.0

_xmin = float(np.min(road_x) - _margin)
_xmax = float(np.max(road_x) + _margin)
_ymin = float(np.min(road_y) - _margin)
_ymax = float(np.max(road_y) + _margin)

_grid_x = np.arange(
    _xmin,
    _xmax + _grid_spacing,
    _grid_spacing,
)
_grid_y = np.arange(
    _ymin,
    _ymax + _grid_spacing,
    _grid_spacing,
)

terrain_x, terrain_y = np.meshgrid(_grid_x, _grid_y)

_grid_lon = terrain_x.ravel() / _m_per_deg_x + _center_lon
_grid_lat = terrain_y.ravel() / _m_per_deg_y + _center_lat

_terrain_cache_key = _cache_key(
    _grid_x,
    _grid_y,
    settings=(
        "QldDem-terrain-v1",
        _grid_spacing,
        _margin,
        _center_lon,
        _center_lat,
        _m_per_deg_x,
        _m_per_deg_y,
    ),
)
_terrain_cache = DATA_DIR / f"terrain_{_terrain_cache_key}.npz"

if _terrain_cache.exists():
    with np.load(_terrain_cache) as cache:
        terrain_z = cache["elevation"].copy()
    if terrain_z.shape != terrain_x.shape:
        raise RuntimeError(f"Invalid terrain elevation cache: {_terrain_cache}")
    print(f"Loaded cached QldDem terrain mesh: {_terrain_cache}")
else:
    print(
        f"Querying QldDem for terrain mesh "
        f"({terrain_x.shape[1]} x {terrain_x.shape[0]} points)..."
    )

    terrain_z = _query_qlddem(
        _grid_lon,
        _grid_lat,
        chunk_size=3,
    ).reshape(terrain_x.shape)
    _save_npz_atomic(_terrain_cache, elevation=terrain_z)


# =====================================================================
# 6. Calculate the coasting speed profile
# =====================================================================

mass = 64.0 + 20.0 + 5.0
gravity = 9.81
air_density = 1.225
rolling_resistance = 0.006
drag_area = 0.50

slope = np.gradient(h, distance)

# Positive sin_theta means downhill acceleration.
sin_theta = np.clip(-slope, -1.0, 1.0)
cos_theta = np.sqrt(np.maximum(0.0, 1.0 - sin_theta**2))

sin_fn = PchipInterpolator(distance, sin_theta, extrapolate=True)
cos_fn = PchipInterpolator(distance, cos_theta, extrapolate=True)


def _speed_ode(s_pos, w):
    st = float(sin_fn(s_pos))
    ct = float(cos_fn(s_pos))
    w0 = max(float(w[0]), 0.0)

    acceleration = (
        gravity * st
        - rolling_resistance * gravity * ct
        - air_density * drag_area / (2 * mass) * w0
    )

    if w0 <= 0.0 and acceleration < 0.0:
        return [0.0]

    return [2.0 * acceleration]


_solution = solve_ivp(
    _speed_ode,
    (float(distance[0]), float(distance[-1])),
    [0.0],
    t_eval=distance,
    rtol=1e-8,
    atol=1e-10,
    max_step=2.0,
)

if not _solution.success:
    raise RuntimeError(f"Speed integration failed: {_solution.message}")

velocity = np.sqrt(np.maximum(_solution.y[0], 0.0))

# Cycling remains possible up to a 5% uphill grade. Interpolate from
# 13 km/h on flat ground to 5 km/h at that threshold; above it, model
# walking speed decreasing from 5 km/h at 5% to 2 km/h at 15% grade.
# The passive coasting model still governs downhill speed.
uphill_grade = np.maximum(slope, 0.0)
ride_grade_limit = 0.05
walk_grade_limit = 0.15
rider_speed_kmh = np.where(
    uphill_grade <= ride_grade_limit,
    13.0 - 8.0 * uphill_grade / ride_grade_limit,
    5.0 - 3.0 * np.clip(
        (uphill_grade - ride_grade_limit)
        / (walk_grade_limit - ride_grade_limit),
        0.0,
        1.0,
    ),
)
rider_powered_mask = slope >= 0.0
velocity[rider_powered_mask] = rider_speed_kmh[rider_powered_mask] / 3.6

print(
    f"Prepared Mount Stuart descent: {distance[-1] / 1000:.3f} km; "
    f"elevation {h[0]:.1f} m -> {h[-1]:.1f} m; "
    f"peak speed {np.max(velocity) * 3.6:.1f} km/h"
)
print("Starting the Mount Stuart animation render...")


# =====================================================================
# 7. Animation configuration
# =====================================================================

OUTPUT_FILE = "mount_stuart_terrain_descent.mp4"

MAX_WORKERS = min(8, os.cpu_count() or 1)
VIDEO_FPS = 10
VIDEO_DPI = 100
VIDEO_CRF = 18
TERRAIN_STRIDE = 3

CAMERA_WIDTH = 700.0
CAMERA_DEPTH = 550.0
CAMERA_LOOK_AHEAD = 100.0
CAMERA_ELEVATION = 38.0
CAMERA_TURN_SMOOTHING = 0.20

LOCAL_Z_BELOW = 65.0
LOCAL_Z_ABOVE = 95.0
VERTICAL_ASPECT_FACTOR = 2.5

TRIKE_SCALE = 4.0
WHEEL_RADIUS = 0.34 * TRIKE_SCALE
REAR_TRACK = 0.85 * TRIKE_SCALE
REAR_OFFSET = -0.85 * TRIKE_SCALE
FRONT_OFFSET = 0.90 * TRIKE_SCALE


# =====================================================================
# 8. Validate and prepare the animation data
# =====================================================================

required = (
    "terrain_x", "terrain_y", "terrain_z",
    "road_x", "road_y", "road_z",
    "distance", "velocity",
)

missing = [name for name in required if name not in globals()]

if missing:
    raise NameError(
        "Missing notebook variables: " + ", ".join(missing)
    )

tx = np.asarray(terrain_x, dtype=float)
ty = np.asarray(terrain_y, dtype=float)
tz = np.asarray(terrain_z, dtype=float)

rx = np.asarray(road_x, dtype=float).ravel()
ry = np.asarray(road_y, dtype=float).ravel()
rz = np.asarray(road_z, dtype=float).ravel()

dist = np.asarray(distance, dtype=float).ravel()
vel = np.asarray(velocity, dtype=float).ravel()

if tx.ndim != 2 or tx.shape != ty.shape or tx.shape != tz.shape:
    raise ValueError(
        "terrain_x, terrain_y and terrain_z must have matching 2D shapes."
    )

if not (len(rx) == len(ry) == len(rz)):
    raise ValueError("Road arrays must have equal lengths.")

if len(rx) < 2 or len(dist) < 2 or len(dist) != len(vel):
    raise ValueError("The route and simulation arrays are incomplete.")

valid = np.isfinite(dist) & np.isfinite(vel)
dist, vel = dist[valid], vel[valid]

order = np.argsort(dist)
dist, vel = dist[order], vel[order]

dist, unique_idx = np.unique(dist, return_index=True)
vel = vel[unique_idx]

if len(dist) < 2 or dist[-1] <= dist[0]:
    raise ValueError("Simulation distance must be strictly increasing.")

if not (
    np.all(np.isfinite(rx))
    and np.all(np.isfinite(ry))
    and np.all(np.isfinite(rz))
):
    raise ValueError("The road arrays contain non-finite coordinates.")

# The upstream route is already summit -> Stuart Drive. Only reverse it
# if its elevations clearly show the opposite orientation.
reverse_route = rz[0] < rz[-1]

if reverse_route:
    rx = rx[::-1].copy()
    ry = ry[::-1].copy()
    rz = rz[::-1].copy()
    print("Reversed road coordinates to match the descent direction.")

route_ds = np.hypot(np.diff(rx), np.diff(ry))
route_s = np.concatenate(([0.0], np.cumsum(route_ds)))

route_s, route_unique_idx = np.unique(
    route_s,
    return_index=True,
)

rx = rx[route_unique_idx]
ry = ry[route_unique_idx]
rz = rz[route_unique_idx]

route_length = float(route_s[-1])
simulation_length = float(dist[-1] - dist[0])

if route_length <= 0 or simulation_length <= 0:
    raise ValueError(
        "Could not determine a positive route or simulation length."
    )

route_progress = (
    (dist - dist[0]) / simulation_length * route_length
)

path_x = np.interp(route_progress, route_s, rx)
path_y = np.interp(route_progress, route_s, ry)
path_z = np.interp(route_progress, route_s, rz)

safe_velocity = np.maximum(vel, 0.1)

# For each segment, use distance divided by the mean endpoint speed. Averaging
# reciprocal speeds overweights a zero-speed endpoint and badly inflates the
# first few elapsed-time intervals.
elapsed = np.zeros_like(dist)
elapsed[1:] = np.cumsum(
    2.0 * np.diff(dist) / (safe_velocity[:-1] + safe_velocity[1:])
)

duration = float(elapsed[-1])

if not np.isfinite(duration) or duration <= 0:
    raise ValueError("Could not calculate a valid simulation duration.")

frame_count = max(
    2,
    int(np.ceil(duration * VIDEO_FPS)) + 1,
)

frame_times = np.linspace(0.0, duration, frame_count)

frame_dist = np.interp(frame_times, elapsed, dist)
frame_vel = np.interp(frame_times, elapsed, vel)
frame_x = np.interp(frame_dist, dist, path_x)
frame_y = np.interp(frame_dist, dist, path_y)
frame_z = np.interp(frame_dist, dist, path_z)

frame_x[0], frame_y[0], frame_z[0] = path_x[0], path_y[0], path_z[0]
frame_x[-1], frame_y[-1], frame_z[-1] = path_x[-1], path_y[-1], path_z[-1]

simulation_output = DATA_DIR / "mount_stuart_simulation.npz"
_save_npz_atomic(
    simulation_output,
    distance=dist,
    elevation=h,
    velocity=vel,
    elapsed_seconds=elapsed,
    route_x=path_x,
    route_y=path_y,
    route_elevation=path_z,
    frame_times=frame_times,
    frame_distance=frame_dist,
    frame_velocity=frame_vel,
    frame_x=frame_x,
    frame_y=frame_y,
    frame_elevation=frame_z,
)
print(f"Saved simulation data: {simulation_output}")

heading_x = np.gradient(frame_x)
heading_y = np.gradient(frame_y)

heading_norm = np.hypot(heading_x, heading_y)
heading_norm[heading_norm < 1e-9] = 1.0

heading_x /= heading_norm
heading_y /= heading_norm

raw_azimuth = np.unwrap(
    np.arctan2(heading_y, heading_x) + np.pi
)

heading_window = max(3, int(round(VIDEO_FPS * 0.8)))

if heading_window % 2 == 0:
    heading_window += 1

heading_kernel = np.ones(heading_window, dtype=float) / heading_window
pad = heading_window // 2

padded_azimuth = np.pad(
    raw_azimuth,
    (pad, pad),
    mode="edge",
)

smoothed_azimuth = np.convolve(
    padded_azimuth,
    heading_kernel,
    mode="valid",
)

camera_azimuth = np.empty_like(smoothed_azimuth)
camera_azimuth[0] = smoothed_azimuth[0]

for frame_i in range(1, len(camera_azimuth)):
    camera_azimuth[frame_i] = (
        camera_azimuth[frame_i - 1]
        + CAMERA_TURN_SMOOTHING
        * (
            smoothed_azimuth[frame_i]
            - camera_azimuth[frame_i - 1]
        )
    )

finite_xy = np.isfinite(tx) & np.isfinite(ty)

terrain_xmin = float(np.min(tx[finite_xy]))
terrain_xmax = float(np.max(tx[finite_xy]))
terrain_ymin = float(np.min(ty[finite_xy]))
terrain_ymax = float(np.max(ty[finite_xy]))

finite_z = tz[np.isfinite(tz)]

if len(finite_z) == 0:
    raise ValueError("The terrain has no finite elevation values.")

# Use the full mesh range for fixed camera clipping. A frame-local z range
# clips the 3D surface differently on each frame and causes large colour
# blocks to flash as the camera moves.
terrain_zmin = float(np.min(finite_z))
terrain_zmax = float(np.max(finite_z))

print(f"Terrain array: {tz.shape}")
print(f"Route length: {route_length:,.1f} m")
print(f"Simulation duration: {duration:,.1f} s")
print(f"Frames: {frame_count:,}")
print(f"Start: ({frame_x[0]:.1f}, {frame_y[0]:.1f}, {frame_z[0]:.1f})")
print(f"Finish: ({frame_x[-1]:.1f}, {frame_y[-1]:.1f}, {frame_z[-1]:.1f})")
print(f"Rendering with up to {MAX_WORKERS} CPU cores.")


_PARALLEL_DATA = {
    "terrain_x": tx[::TERRAIN_STRIDE, ::TERRAIN_STRIDE],
    "terrain_y": ty[::TERRAIN_STRIDE, ::TERRAIN_STRIDE],
    "terrain_z": tz[::TERRAIN_STRIDE, ::TERRAIN_STRIDE],
    "road_x": rx,
    "road_y": ry,
    "road_z": rz,
    "times": frame_times,
    "distance": frame_dist,
    "velocity": frame_vel,
    "x": frame_x,
    "y": frame_y,
    "z": frame_z,
    "hx": heading_x,
    "hy": heading_y,
    "camera_azimuth": camera_azimuth,
    "frame_count": frame_count,
    "terrain_xmin": terrain_xmin,
    "terrain_xmax": terrain_xmax,
    "terrain_ymin": terrain_ymin,
    "terrain_ymax": terrain_ymax,
    "terrain_zmin": terrain_zmin,
    "terrain_zmax": terrain_zmax,
    "camera_width": CAMERA_WIDTH,
    "camera_depth": CAMERA_DEPTH,
    "look_ahead": CAMERA_LOOK_AHEAD,
    "camera_elevation": CAMERA_ELEVATION,
    "local_z_below": LOCAL_Z_BELOW,
    "local_z_above": LOCAL_Z_ABOVE,
    "vertical_aspect_factor": VERTICAL_ASPECT_FACTOR,
    "wheel_radius": WHEEL_RADIUS,
    "rear_track": REAR_TRACK,
    "rear_offset": REAR_OFFSET,
    "front_offset": FRONT_OFFSET,
    "video_dpi": VIDEO_DPI,
}

_WORKER_SCENE = None


# =====================================================================
# 9. Initialize a Matplotlib scene in each worker
# =====================================================================

def _initialize_render_worker():
    global _WORKER_SCENE

    d = _PARALLEL_DATA

    fig = plt.figure(figsize=(13.0, 9.0), dpi=d["video_dpi"])
    ax = fig.add_subplot(111, projection="3d")

    zz = d["terrain_z"]

    zmin = float(d["terrain_zmin"])
    zmax = float(d["terrain_zmax"])

    if zmax <= zmin:
        zmax = zmin + 1.0

    terrain_norm = mpl.colors.Normalize(vmin=zmin, vmax=zmax, clip=True)
    ax.plot_surface(
        d["terrain_x"],
        d["terrain_y"],
        zz,
        cmap=cm.terrain,
        norm=terrain_norm,
        linewidth=0,
        antialiased=False,
        # Map each polygon from its elevation with one fixed normalization;
        # avoid view-dependent lighting that can make the surface flash.
        shade=False,
    )

    colorbar_ax = fig.add_axes([0.91, 0.22, 0.022, 0.55])
    colorbar = fig.colorbar(
        cm.ScalarMappable(norm=terrain_norm, cmap=cm.terrain),
        cax=colorbar_ax,
    )
    colorbar.set_label("Elevation (m)")

    ax.plot(
        d["road_x"],
        d["road_y"],
        d["road_z"] + 0.8,
        color="black",
        linewidth=3.2,
        zorder=15,
    )

    ax.plot(
        d["road_x"],
        d["road_y"],
        d["road_z"] + 1.6,
        color="white",
        linewidth=1.1,
        zorder=16,
    )

    trail, = ax.plot(
        [],
        [],
        [],
        color="orangered",
        linewidth=3.0,
        zorder=25,
    )

    names = [
        "rear_axle", "front_fork", "main_frame", "seat_stay",
        "handlebar", "crank", "seat", "torso", "arms", "legs",
    ]

    segments = {}

    for name in names:
        colour = "#ff8c00"
        width = 4.5

        if name in ("torso", "arms", "legs"):
            colour = "#b21f24"
            width = 5.5

        line, = ax.plot(
            [],
            [],
            [],
            color=colour,
            linewidth=width,
            solid_capstyle="round",
            zorder=40,
        )

        segments[name] = line

    wheels, hubs = [], []

    for _ in range(3):
        wheel, = ax.plot(
            [],
            [],
            [],
            color="#111111",
            linewidth=3.5,
            zorder=42,
        )
        wheels.append(wheel)

        hub, = ax.plot(
            [],
            [],
            [],
            marker="o",
            markersize=5,
            linestyle="None",
            color="white",
            markeredgecolor="black",
            markeredgewidth=1.2,
            zorder=44,
        )
        hubs.append(hub)

    head, = ax.plot(
        [],
        [],
        [],
        marker="o",
        markersize=11,
        linestyle="None",
        markerfacecolor="#ffd28a",
        markeredgecolor="black",
        markeredgewidth=1.5,
        zorder=48,
    )

    fig.text(
        0.5,
        0.975,
        "Mount Stuart Road descent",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
    )

    status = fig.text(
        0.5,
        0.925,
        "",
        ha="center",
        va="top",
        fontsize=11,
    )

    fig.subplots_adjust(
        left=0.04,
        right=0.88,
        bottom=0.12,
        top=0.87,
    )

    ax.set_axis_off()
    ax.set_facecolor("#eeeeee")
    ax.set_proj_type("persp", focal_length=0.9)
    ax.set_box_aspect((
        d["camera_width"],
        d["camera_depth"],
        max(
            (zmax - zmin) * d["vertical_aspect_factor"],
            1.0,
        ),
    ))

    _WORKER_SCENE = {
        "fig": fig,
        "ax": ax,
        "trail": trail,
        "segments": segments,
        "wheels": wheels,
        "hubs": hubs,
        "head": head,
        "status": status,
    }


# =====================================================================
# 10. Render an individual animation frame
# =====================================================================

def _render_one_frame(i, output_directory):
    global _WORKER_SCENE

    if _WORKER_SCENE is None:
        _initialize_render_worker()

    d = _PARALLEL_DATA
    scene = _WORKER_SCENE
    ax = scene["ax"]

    x = float(d["x"][i])
    y = float(d["y"][i])
    z = float(d["z"][i])

    forward = np.array(
        [d["hx"][i], d["hy"][i]],
        dtype=float,
    )

    n = np.linalg.norm(forward)

    if n > 1e-9:
        forward /= n
    else:
        forward = np.array([1.0, 0.0])

    sideways = np.array([-forward[1], forward[0]])

    def p(longitudinal, lateral=0.0, height=0.0):
        return np.array([
            x + longitudinal * forward[0] + lateral * sideways[0],
            y + longitudinal * forward[1] + lateral * sideways[1],
            z + height,
        ])

    r = d["wheel_radius"]
    track = d["rear_track"]

    rear_left = p(d["rear_offset"], -track / 2, r)
    rear_right = p(d["rear_offset"], track / 2, r)
    front = p(d["front_offset"], 0.0, r)
    rear_mid = (rear_left + rear_right) / 2.0

    theta = np.linspace(0, 2 * np.pi, 48)

    for wheel, hub, centre in zip(
        scene["wheels"],
        scene["hubs"],
        [rear_left, rear_right, front],
    ):
        wheel.set_data_3d(
            centre[0] + r * np.cos(theta) * forward[0],
            centre[1] + r * np.cos(theta) * forward[1],
            centre[2] + r * np.sin(theta),
        )

        hub.set_data_3d(
            [centre[0]],
            [centre[1]],
            [centre[2]],
        )

    crank = p(
        -0.05 * TRIKE_SCALE,
        0,
        0.65 * TRIKE_SCALE,
    )

    seat = p(
        -0.42 * TRIKE_SCALE,
        0,
        1.02 * TRIKE_SCALE,
    )

    head_tube = p(
        0.40 * TRIKE_SCALE,
        0,
        0.95 * TRIKE_SCALE,
    )

    handlebar = p(
        0.48 * TRIKE_SCALE,
        0,
        1.10 * TRIKE_SCALE,
    )

    geometry = {
        "rear_axle": [rear_left, rear_right],
        "front_fork": [head_tube, front],
        "main_frame": [rear_mid, crank, head_tube],
        "seat_stay": [rear_mid, seat, crank],
        "handlebar": [head_tube, handlebar],
        "crank": [
            p(-0.18 * TRIKE_SCALE, 0, 0.65 * TRIKE_SCALE),
            p(0.08 * TRIKE_SCALE, 0, 0.65 * TRIKE_SCALE),
        ],
        "seat": [
            p(-0.55 * TRIKE_SCALE, 0, 1.02 * TRIKE_SCALE),
            p(-0.30 * TRIKE_SCALE, 0, 1.02 * TRIKE_SCALE),
        ],
    }

    hip = p(-0.38 * TRIKE_SCALE, 0, 1.30 * TRIKE_SCALE)
    shoulder = p(-0.10 * TRIKE_SCALE, 0, 1.80 * TRIKE_SCALE)
    head_centre = p(-0.08 * TRIKE_SCALE, 0, 2.02 * TRIKE_SCALE)
    hand = p(0.42 * TRIKE_SCALE, 0, 1.12 * TRIKE_SCALE)
    knee = p(-0.05 * TRIKE_SCALE, 0, 1.02 * TRIKE_SCALE)
    foot = p(0.02 * TRIKE_SCALE, 0, 0.65 * TRIKE_SCALE)

    geometry["torso"] = [hip, shoulder]
    geometry["arms"] = [shoulder, hand]
    geometry["legs"] = [hip, knee, foot]

    for name, points in geometry.items():
        arr = np.asarray(points)

        scene["segments"][name].set_data_3d(
            arr[:, 0],
            arr[:, 1],
            arr[:, 2],
        )

    scene["head"].set_data_3d(
        [head_centre[0]],
        [head_centre[1]],
        [head_centre[2]],
    )

    trail_start = max(0, i - VIDEO_FPS * 8)

    scene["trail"].set_data_3d(
        d["x"][trail_start:i + 1],
        d["y"][trail_start:i + 1],
        d["z"][trail_start:i + 1] + 1.5,
    )

    azimuth = float(np.degrees(d["camera_azimuth"][i]))

    ax.view_init(
        elev=d["camera_elevation"],
        azim=azimuth,
    )

    cx = x + forward[0] * d["look_ahead"]
    cy = y + forward[1] * d["look_ahead"]

    half_w = d["camera_width"] / 2.0
    half_h = d["camera_depth"] / 2.0

    x0, x1 = cx - half_w, cx + half_w
    y0, y1 = cy - half_h, cy + half_h

    if x0 < d["terrain_xmin"]:
        x1 += d["terrain_xmin"] - x0
        x0 = d["terrain_xmin"]

    if x1 > d["terrain_xmax"]:
        x0 -= x1 - d["terrain_xmax"]
        x1 = d["terrain_xmax"]

    if y0 < d["terrain_ymin"]:
        y1 += d["terrain_ymin"] - y0
        y0 = d["terrain_ymin"]

    if y1 > d["terrain_ymax"]:
        y0 -= y1 - d["terrain_ymax"]
        y1 = d["terrain_ymax"]

    x0 = max(x0, d["terrain_xmin"])
    x1 = min(x1, d["terrain_xmax"])
    y0 = max(y0, d["terrain_ymin"])
    y1 = min(y1, d["terrain_ymax"])

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    # Keep every terrain vertex in view on every frame. Changing z-limits
    # locally clips the Poly3DCollection and caused the terrain-wide flashes.
    ax.set_zlim(d["terrain_zmin"], d["terrain_zmax"])

    t = float(d["times"][i])
    v = float(d["velocity"][i])
    s_value = float(d["distance"][i])

    whole_seconds = int(max(t, 0.0))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_parts = []

    if hours:
        time_parts.append(f"{hours} h")

    if minutes:
        time_parts.append(f"{minutes} min")

    seconds_value = max(t, 0.0) - hours * 3600 - minutes * 60
    time_parts.append(f"{seconds_value:.1f} s")
    readable_time = " ".join(time_parts)

    scene["status"].set_text(
        f"Elapsed: {readable_time}"
        f"     |     Velocity: {v:,.2f} m/s ({v * 3.6:,.1f} km/h)"
        f"     |     Distance: {s_value:,.0f} m"
        f"     |     Elevation: {z:,.1f} m"
    )

    frame_path = os.path.join(
        output_directory,
        f"frame_{i:06d}.jpg",
    )

    scene["fig"].savefig(
        frame_path,
        dpi=d["video_dpi"],
        format="jpg",
        pil_kwargs={"quality": 92},
    )

    return int(i)


# =====================================================================
# 11. Render frames in parallel and encode the MP4
# =====================================================================

ffmpeg_path = shutil.which("ffmpeg")

if ffmpeg_path is None:
    raise RuntimeError(
        "FFmpeg was not found. Install it and ensure ffmpeg is on PATH."
    )

if "fork" not in mp.get_all_start_methods():
    raise RuntimeError(
        "This cell requires Linux multiprocessing with the fork start method."
    )

with tempfile.TemporaryDirectory(
    prefix="mount_stuart_frames_"
) as frame_dir:

    context = mp.get_context("fork")
    completed = 0
    last_line_length = 0

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=context,
        initializer=_initialize_render_worker,
    ) as executor:

        futures = {
            executor.submit(
                _render_one_frame,
                i,
                frame_dir,
            ): i
            for i in range(frame_count)
        }

        for future in as_completed(futures):
            finished_frame = future.result()
            completed += 1
            percentage = 100.0 * completed / frame_count

            message = (
                f"Completed frame {finished_frame + 1:,}/{frame_count:,}"
                f" | Overall progress: {percentage:6.2f}%"
                f" | Frames finished: {completed:,}/{frame_count:,}"
            )

            print(
                "\r" + message
                + " " * max(0, last_line_length - len(message)),
                end="",
                flush=True,
            )

            last_line_length = len(message)

    print("\nAll frames rendered. Encoding MP4...")

    subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel", "warning",
            "-framerate", str(VIDEO_FPS),
            "-i", os.path.join(frame_dir, "frame_%06d.jpg"),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", str(VIDEO_CRF),
            "-pix_fmt", "yuv420p",
            "-threads", str(MAX_WORKERS),
            "-movflags", "+faststart",
            OUTPUT_FILE,
        ],
        check=True,
    )

print(f"\nFinished: {os.path.abspath(OUTPUT_FILE)}")
print(f"File size: {os.path.getsize(OUTPUT_FILE) / (1024 ** 2):,.1f} MiB")
