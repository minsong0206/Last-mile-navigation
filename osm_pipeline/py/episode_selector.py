"""
Step 1: Episode Selection via EKF-trajectory vs OSRM-route alignment

Refinements over v1:
  1. Stationary-frame removal — trim leading/trailing stationary frames and
     strip interior stationary frames so only moving frames are evaluated.
  2. Segment splitting — long pauses (consecutive stationary frames) indicate
     the robot restarted from a new location. Each contiguous moving segment
     is evaluated independently and can be selected as its own sub-episode.
  3. Sidewalk validation — OSRM (foot profile) nearest-road snap distance
     must be below threshold, confirming the trajectory lies on pedestrian paths.

OSRM servers must be running:
  Perth      → localhost:5001
  Taipei     → localhost:5002
  Tokyo      → localhost:5003
  Wuhan      → localhost:5004
  Manila     → localhost:5005
  Rome       → localhost:5006
  Wellington → localhost:5007
  Florida    → localhost:5008
  Brighton   → localhost:5009
  Madrid     → localhost:5010

Output: episode_scores.json
  Each entry has an 'episode' and optional 'segment' field.
  Sub-episodes from the same raw episode share the same 'episode' value
  but differ in 'segment' (0, 1, 2, …).
"""

import os
import json
import argparse
import math
import numpy as np
import pyarrow as pa
import requests
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

ARROW_PATH = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/FrodoBots-2K/processed/output_rides_11/train/data-00000-of-00001.arrow"
OUT_DIR    = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/output_rides_11"

# ── OSRM port routing ────────────────────────────────────────────────────────
def osrm_port(lat, lon):
    if lat < -30:                            return 5001  # Perth
    if 24 < lat < 26 and 121 < lon < 122:   return 5002  # Taipei
    if 35 < lat < 36 and 139 < lon < 140:   return 5003  # Tokyo
    if 30 < lat < 31 and 114 < lon < 115:   return 5004  # Wuhan
    if 14 < lat < 15 and 120 < lon < 122:   return 5005  # Manila
    if 41 < lat < 43 and 12 < lon < 13:     return 5006  # Rome
    if -42 < lat < -41 and 174 < lon < 175: return 5007  # Wellington
    if 27 < lat < 28 and -81 < lon < -80:   return 5008  # Florida
    if 50 < lat < 51 and -1 < lon < 0:      return 5009  # Brighton
    if 40 < lat < 41 and -4 < lon < -3:     return 5010  # Madrid
    return None

# ── Thresholds ───────────────────────────────────────────────────────────────
MIN_FRAMES         = 100       # minimum moving frames after stripping stationary
MIN_TRAJ_LEN_M     = 30.0     # minimum trajectory length in metres
MAX_FRECHET_NORM   = 0.5      # Fréchet / traj_length
MAX_CHAMFER_NORM   = 0.15     # Chamfer / traj_length
MAX_HEADING_ERR    = 45.0     # degrees
LENGTH_RATIO_MIN   = 0.5
LENGTH_RATIO_MAX   = 2.0

# Stationary / segment parameters
STAT_DIST_M        = 0.05     # frame-to-frame dist (m) below which frame is stationary
LONG_STOP_FRAMES   = 50       # consecutive stationary frames → segment boundary
MAX_SNAP_DIST_M    = 15.0     # max OSRM nearest-road snap distance (sidewalk check)


# ── UTM conversion ────────────────────────────────────────────────────────────

def latlon_to_utm_xy(lats, lons):
    import utm
    zone_num = None
    x0 = y0 = 0.0
    xys = []
    for lat, lon in zip(lats, lons):
        x, y, zn, _ = utm.from_latlon(lat, lon)
        if zone_num is None:
            zone_num, x0, y0 = zn, x, y
        xys.append([x - x0, y - y0])
    return np.array(xys)


def route_length(pts):
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


# ── Stationary-frame removal & segment detection ──────────────────────────────

def split_into_segments(fp, lats, lons):
    """
    Remove stationary frames and split at long stops.

    Returns list of dicts, each with keys:
      'fp', 'lats', 'lons' — arrays for one contiguous moving segment
      'frame_indices'       — original frame indices within the episode
    """
    if len(fp) < 2:
        return []

    # Frame-to-frame distance (len-1 array; frame 0 gets distance from frame 1)
    dists = np.linalg.norm(np.diff(fp, axis=0), axis=1)
    # Pad to same length as fp: frame 0 is moving if frame 1 is far enough
    is_moving = np.concatenate([[dists[0] >= STAT_DIST_M], dists >= STAT_DIST_M])

    # Identify segment boundaries: runs of >= LONG_STOP_FRAMES consecutive stationary frames
    # Mark positions where a long stop occurs
    stationary = ~is_moving
    seg_break = np.zeros(len(fp), dtype=bool)

    run_start = None
    for i in range(len(fp)):
        if stationary[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_len = i - run_start
                if run_len >= LONG_STOP_FRAMES:
                    seg_break[run_start:i] = True
                run_start = None
    if run_start is not None and len(fp) - run_start >= LONG_STOP_FRAMES:
        seg_break[run_start:] = True

    # Build segments: contiguous runs of frames that are moving AND not at a seg boundary
    keep = is_moving & ~seg_break
    segments = []
    in_seg = False
    seg_start = 0
    for i in range(len(fp)):
        if keep[i] and not in_seg:
            seg_start = i
            in_seg = True
        elif not keep[i] and in_seg:
            idxs = np.where(keep[seg_start:i])[0] + seg_start
            # idxs are positions within the episode frame array that we keep
            # But we want only the consecutive moving run, not gaps
            idxs = np.arange(seg_start, i)[keep[seg_start:i]]
            if len(idxs) > 0:
                segments.append({
                    'fp':           fp[idxs],
                    'lats':         lats[idxs],
                    'lons':         lons[idxs],
                    'frame_indices': idxs,
                })
            in_seg = False

    if in_seg:
        idxs = np.arange(seg_start, len(fp))[keep[seg_start:]]
        if len(idxs) > 0:
            segments.append({
                'fp':           fp[idxs],
                'lats':         lats[idxs],
                'lons':         lons[idxs],
                'frame_indices': idxs,
            })

    return segments


# ── OSRM route via waypoints ──────────────────────────────────────────────────

def osrm_route_waypoints(lats, lons, port, n_waypoints=8):
    idxs = np.linspace(0, len(lats) - 1, min(n_waypoints, len(lats)), dtype=int)
    coords_str = ";".join(f"{lons[i]},{lats[i]}" for i in idxs)
    url = (f"http://localhost:{port}/route/v1/foot/{coords_str}"
           f"?overview=full&geometries=geojson")
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        d = r.json()
        if d.get("code") != "Ok" or not d.get("routes"):
            return None
        return d["routes"][0]["geometry"]["coordinates"]
    except Exception:
        return None


def osrm_coords_to_utm_xy(coords, ref_lat, ref_lon):
    import utm
    _, _, zone_num, _ = utm.from_latlon(ref_lat, ref_lon)
    ref_x, ref_y, _, _ = utm.from_latlon(ref_lat, ref_lon)
    pts = []
    for lon, lat in coords:
        x, y, _, _ = utm.from_latlon(lat, lon, force_zone_number=zone_num)
        pts.append([x - ref_x, y - ref_y])
    return np.array(pts)


# ── Sidewalk snap check via OSRM nearest ──────────────────────────────────────

def check_sidewalk_snap(lats, lons, port, n_check=10, max_snap_m=MAX_SNAP_DIST_M):
    """
    Sample n_check evenly-spaced trajectory points, snap each to the nearest
    OSRM foot-profile road, and check that no snap exceeds max_snap_m metres.
    Returns (ok: bool, max_snap_found: float).
    """
    idxs = np.linspace(0, len(lats) - 1, min(n_check, len(lats)), dtype=int)
    max_snap = 0.0
    for i in idxs:
        url = (f"http://localhost:{port}/nearest/v1/foot/"
               f"{lons[i]},{lats[i]}?number=1")
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            d = r.json()
            if d.get("code") != "Ok":
                return False, 9999.0
            # OSRM returns distance in metres
            dist = d["waypoints"][0].get("distance", 9999.0)
            max_snap = max(max_snap, dist)
        except Exception:
            return False, 9999.0
    return max_snap <= max_snap_m, max_snap


# ── Path comparison metrics ───────────────────────────────────────────────────

def resample_path(pts, n=100):
    dists = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    if dists[-1] < 1e-3:
        return pts
    new_dists = np.linspace(0, dists[-1], n)
    xs = np.interp(new_dists, dists, pts[:, 0])
    ys = np.interp(new_dists, dists, pts[:, 1])
    return np.stack([xs, ys], axis=1)


def frechet_distance(p, q):
    """Discrete Fréchet distance via iterative DP (avoids recursion limit)."""
    n, m = len(p), len(q)
    ca = np.full((n, m), np.inf)
    ca[0, 0] = np.linalg.norm(p[0] - q[0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j-1], np.linalg.norm(p[0] - q[j]))
    for i in range(1, n):
        ca[i, 0] = max(ca[i-1, 0], np.linalg.norm(p[i] - q[0]))
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(min(ca[i-1, j], ca[i-1, j-1], ca[i, j-1]),
                           np.linalg.norm(p[i] - q[j]))
    return float(ca[n-1, m-1])


def chamfer_distance(p, q):
    from scipy.spatial import cKDTree
    d_pq, _ = cKDTree(q).query(p)
    d_qp, _ = cKDTree(p).query(q)
    return float((d_pq.mean() + d_qp.mean()) / 2.0)


def mean_heading_error(p, q):
    def headings(pts):
        diffs = np.diff(pts, axis=0)
        return np.degrees(np.arctan2(diffs[:, 1], diffs[:, 0]))
    h_p = headings(p)
    h_q = headings(q)
    n = min(len(h_p), len(h_q))
    diff = np.abs(h_p[:n] - h_q[:n])
    diff = np.minimum(diff, 360 - diff)
    return float(diff.mean())


# ── Per-segment scoring ───────────────────────────────────────────────────────

def score_segment(ep, seg_idx, seg):
    lats = seg['lats']
    lons = seg['lons']
    fp   = seg['fp']

    n_frames  = len(lats)
    traj_len  = route_length(fp)

    result = {
        'episode':       ep,
        'segment':       seg_idx,
        'n_frames':      int(n_frames),
        'traj_len_m':    float(traj_len),
        'selected':      False,
        'reject_reason': None,
    }

    if n_frames < MIN_FRAMES:
        result['reject_reason'] = f'too_short_frames({n_frames})'
        return result
    if traj_len < MIN_TRAJ_LEN_M:
        result['reject_reason'] = f'too_short_traj({traj_len:.1f}m)'
        return result

    port = osrm_port(float(np.mean(lats)), float(np.mean(lons)))
    if port is None:
        result['reject_reason'] = 'no_osrm_server'
        return result

    # ── Sidewalk snap check ───────────────────────────────────────────────────
    on_sidewalk, max_snap = check_sidewalk_snap(lats, lons, port)
    result['max_snap_m'] = float(max_snap)
    if not on_sidewalk:
        result['reject_reason'] = f'off_sidewalk(snap={max_snap:.1f}m>{MAX_SNAP_DIST_M}m)'
        return result

    # ── OSRM route (multi-waypoint to handle curves) ──────────────────────────
    coords = osrm_route_waypoints(lats, lons, port, n_waypoints=8)
    if coords is None or len(coords) < 2:
        result['reject_reason'] = 'osrm_route_failed'
        return result

    osm_path = osrm_coords_to_utm_xy(coords, lats[0], lons[0])
    osm_len  = route_length(osm_path)
    length_ratio = osm_len / (traj_len + 1e-3)

    # ── Resample & align origins ──────────────────────────────────────────────
    ekf_rs = resample_path(fp,       100) - fp[0]
    osm_rs = resample_path(osm_path, 100) - osm_path[0]

    fd       = frechet_distance(ekf_rs, osm_rs)
    cd       = chamfer_distance(ekf_rs, osm_rs)
    head_err = mean_heading_error(ekf_rs, osm_rs)
    fd_norm  = fd / (traj_len + 1e-3)
    cd_norm  = cd / (traj_len + 1e-3)

    result.update({
        'osm_len_m':       float(osm_len),
        'length_ratio':    float(length_ratio),
        'frechet_norm':    float(fd_norm),
        'chamfer_norm':    float(cd_norm),
        'heading_err_deg': float(head_err),
    })

    reasons = []
    if fd_norm > MAX_FRECHET_NORM:
        reasons.append(f'frechet={fd_norm:.3f}>{MAX_FRECHET_NORM}')
    if cd_norm > MAX_CHAMFER_NORM:
        reasons.append(f'chamfer={cd_norm:.3f}>{MAX_CHAMFER_NORM}')
    if head_err > MAX_HEADING_ERR:
        reasons.append(f'heading={head_err:.1f}>{MAX_HEADING_ERR}')
    if not (LENGTH_RATIO_MIN <= length_ratio <= LENGTH_RATIO_MAX):
        reasons.append(f'len_ratio={length_ratio:.2f}')

    if reasons:
        result['reject_reason'] = '; '.join(reasons)
    else:
        result['selected'] = True

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "episode_scores.json")

    print("Loading dataset...")
    table  = pa.ipc.open_stream(open(ARROW_PATH, 'rb')).read_all()
    ep_idx = np.array(table['episode_index'].to_pylist())
    lats   = np.array(table['observation.latitude'].to_pylist())
    lons   = np.array(table['observation.longitude'].to_pylist())
    fp     = np.array(table['observation.filtered_position'].to_pylist())
    total_eps = int(ep_idx.max()) + 1
    print(f"Total episodes: {total_eps}")

    # Resume support — keyed on (episode, segment)
    if os.path.exists(out_path):
        with open(out_path) as f:
            scores = json.load(f)
        done_keys = {(s['episode'], s.get('segment', 0)) for s in scores}
        # Determine which raw episodes are fully processed:
        # an episode is "done" only if we won't re-split it differently.
        # We re-run an episode if its raw episode key is not represented at all.
        done_eps = {s['episode'] for s in scores}
        print(f"Resuming: {len(done_eps)} episodes already scored "
              f"({len(scores)} segments total)")
    else:
        scores, done_eps = [], set()

    ep_range = range(args.ep_start, min(args.ep_end, total_eps))
    for ep in tqdm(ep_range, desc="Scoring episodes"):
        if ep in done_eps:
            continue

        mask = ep_idx == ep
        ep_fp   = fp[mask]
        ep_lats = lats[mask]
        ep_lons = lons[mask]

        # Split into contiguous moving segments
        segments = split_into_segments(ep_fp, ep_lats, ep_lons)

        if not segments:
            scores.append({
                'episode':       ep,
                'segment':       0,
                'n_frames':      int(mask.sum()),
                'traj_len_m':    0.0,
                'selected':      False,
                'reject_reason': 'no_moving_frames',
            })
        else:
            for seg_idx, seg in enumerate(segments):
                result = score_segment(ep, seg_idx, seg)
                scores.append(result)
                status = ("✓ SELECTED" if result['selected']
                          else f"✗ {result.get('reject_reason', '')}")
                tqdm.write(
                    f"  ep{ep:03d}[{seg_idx}]: n={result['n_frames']:5d} "
                    f"traj={result.get('traj_len_m', 0):6.0f}m  {status}"
                )

        with open(out_path, 'w') as f:
            json.dump(scores, f, indent=2)

    selected = [s for s in scores if s['selected']]
    print(f"\n=== Done ===")
    print(f"Segments scored: {len(scores)}  Selected: {len(selected)}/{len(scores)}")
    print(f"Results → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep_start", type=int, default=0)
    parser.add_argument("--ep_end",   type=int, default=9999)
    args = parser.parse_args()
    main(args)
