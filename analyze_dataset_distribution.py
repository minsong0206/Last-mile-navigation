"""
analyze_dataset_distribution.py

Analyzes scenario distribution of the rides_11 dataset by computing
trajectory statistics from GPS data for each selected segment.

Classifies each segment into:
  - straight          : low total curvature, minimal heading change
  - curve_left        : gradual net left turn
  - curve_right       : gradual net right turn
  - sharp_left_turn   : rapid concentrated left turn (>60 deg in short distance)
  - sharp_right_turn  : rapid concentrated right turn (>60 deg in short distance)
  - s_curve           : significant heading change in BOTH directions
  - intersection      : multiple distinct turn events in one segment
  - complex_urban     : high curvature density, many direction reversals

Also produces:
  - dataset_labels.json  : per-segment classification + stats
  - distribution plots   : scenario pie, curvature histogram, trajectory maps

Run on host (no conda needed):
  cd /media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA
  python3 analyze_dataset_distribution.py
"""

import os
import csv
import json
import math
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict, Counter

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path("/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA")
OSM_ROOT    = BASE / "osm_pipeline/osm_data/output_rides_11/osm_maps_arrow"
SCORES_PATH = BASE / "osm_pipeline/osm_data/output_rides_11/episode_scores.json"
OUT_DIR     = BASE / "dataset_analysis"
OUT_DIR.mkdir(exist_ok=True)

# ── Thresholds ─────────────────────────────────────────────────────────────────
# GPS from FrodoBots has ~0.09m step size with significant noise.
# We subsample to DIST_STEP_M intervals before computing headings.
DIST_STEP_M         = 2.0  # meters: resample GPS to suppress sub-meter noise
SMOOTH_WINDOW       = 3    # waypoints for heading smoothing after resampling
TURN_EVENT_THRESH   = 30   # deg: minimum accumulation to count as a distinct turn event
SHARP_TURN_THRESH   = 45   # deg: single event reaching this = "sharp" turn
NET_STRAIGHT_THRESH = 20   # deg: |net heading| below this → straight candidate
TOTAL_STRAIGHT_THRESH = 100 # deg: total curvature below this → straight candidate
CURV_DENSITY_THRESH = 800  # deg/100m: above p75 (414) → very dense campus winding

# ── Utility: GPS → local XY (meters) ─────────────────────────────────────────
R_EARTH = 6_371_000.0  # meters

def latlon_to_xy(lats, lons):
    """Convert arrays of lat/lon to local XY using equirectangular projection."""
    lat0, lon0 = lats[0], lons[0]
    x = np.radians(lons - lon0) * math.cos(math.radians(lat0)) * R_EARTH
    y = np.radians(lats - lat0) * R_EARTH
    return x, y


def smooth(arr, w):
    """Simple box-filter smoothing with edge padding."""
    if w <= 1 or len(arr) < w:
        return arr.copy()
    kernel = np.ones(w) / w
    pad = w // 2
    padded = np.pad(arr, pad, mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(arr)]


def angle_diff(a, b):
    """Signed difference b - a, wrapped to (-180, 180]."""
    d = (b - a + 180) % 360 - 180
    return d


def resample_by_distance(x, y, step_m=DIST_STEP_M):
    """
    Resample an XY trajectory at uniform distance intervals.
    Suppresses GPS noise: each step now represents `step_m` meters of movement,
    so the heading signal is computed from a meaningful baseline, not sub-meter jitter.
    Returns resampled (rx, ry) arrays.
    """
    dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    cum_dist = np.concatenate([[0.0], np.cumsum(dists)])
    total = cum_dist[-1]
    if total < step_m:
        return x, y  # segment too short to resample
    targets = np.arange(0, total, step_m)
    rx = np.interp(targets, cum_dist, x)
    ry = np.interp(targets, cum_dist, y)
    return rx, ry


def compute_headings(x, y):
    """Per-waypoint heading in degrees from resampled XY trajectory."""
    dx = np.diff(x)
    dy = np.diff(y)
    headings = np.degrees(np.arctan2(dy, dx))  # shape: N-1
    headings = np.concatenate([[headings[0]], headings])
    return headings


def compute_heading_changes(headings, smooth_w=SMOOTH_WINDOW):
    """Returns array of signed heading changes (deg), after smoothing."""
    s = smooth(headings, smooth_w)
    changes = np.array([angle_diff(s[i], s[i+1]) for i in range(len(s)-1)])
    return changes


def detect_turn_events(changes, thresh=TURN_EVENT_THRESH):
    """
    Finds contiguous runs where the heading is consistently changing in one
    direction and the total accumulated magnitude exceeds `thresh`.

    Returns list of dicts: {start, end, total_deg, direction ('L'/'R')}
    """
    events = []
    n = len(changes)
    i = 0
    while i < n:
        if abs(changes[i]) < 0.5:   # near-zero step, skip
            i += 1
            continue
        sign = 1 if changes[i] > 0 else -1
        acc = changes[i]
        j = i + 1
        while j < n:
            if changes[j] * sign > 0:   # same direction
                acc += changes[j]
            elif abs(changes[j]) > 2.0:  # significant reversal → stop
                break
            j += 1
        magnitude = abs(acc)
        if magnitude >= thresh:
            events.append({
                "start": i, "end": j,
                "total_deg": acc,
                "direction": "L" if acc > 0 else "R",
            })
        i = j if j > i else i + 1
    return events


def classify_segment(stats):
    """
    Maps per-segment stats to a scenario label.
    Calibrated for slow campus robot GPS (FrodoBots rides_11):
      - 2m-resampled trajectory, ~414 deg/100m median curvature density

    Priority (high → low specificity):
      1. complex_urban    – density > 800 deg/100m AND 4+ turn events
      2. winding          – 4+ turn events (lower density, campus paths)
      3. intersection     – 2+ events in BOTH L and R (incl. S-curves, junctions)
      4. sharp_left_turn  – concentrated single left event ≥ 45°
      5. sharp_right_turn – concentrated single right event ≥ 45°
      6. curve_left       – gradual net left turn, few events
      7. curve_right      – gradual net right turn, few events
      8. straight         – minimal net and total heading change
    """
    n_events   = stats["n_turn_events"]
    n_L        = stats["n_left_events"]
    n_R        = stats["n_right_events"]
    max_L      = stats["max_left_turn_deg"]
    max_R      = stats["max_right_turn_deg"]
    net        = stats["net_heading_deg"]
    total      = stats["total_heading_deg"]
    curv_den   = stats["curvature_density"]

    # complex urban: very dense winding AND many turn events (top ~25%)
    if curv_den > CURV_DENSITY_THRESH and n_events >= 4:
        return "complex_urban"

    # winding: many turn events (campus curves) but lower density
    if n_events >= 4:
        return "winding"

    # intersection / S-curve: turned in BOTH directions (requires navigation decision)
    if n_events >= 2 and n_L >= 1 and n_R >= 1:
        return "intersection"

    # sharp single turns
    if max_L >= SHARP_TURN_THRESH and n_L >= 1:
        return "sharp_left_turn"
    if max_R >= SHARP_TURN_THRESH and n_R >= 1:
        return "sharp_right_turn"

    # gradual single-direction curves
    if net > NET_STRAIGHT_THRESH and total > TOTAL_STRAIGHT_THRESH:
        return "curve_left"
    if net < -NET_STRAIGHT_THRESH and total > TOTAL_STRAIGHT_THRESH:
        return "curve_right"

    # straight (may have minor GPS noise wiggles)
    return "straight"


# ── Main analysis ─────────────────────────────────────────────────────────────

def load_selected_segments():
    with open(SCORES_PATH) as f:
        scores = json.load(f)
    return [s for s in scores if s["selected"]]


def read_gps(seg_dir):
    path = seg_dir / "gps.csv"
    if not path.exists():
        return None, None
    lats, lons = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            lats.append(float(row["latitude"]))
            lons.append(float(row["longitude"]))
    return np.array(lats), np.array(lons)


def analyze_segment(seg_info):
    ep  = seg_info["episode"]
    seg = seg_info["segment"]
    seg_dir = OSM_ROOT / f"episode_{ep:04d}_seg{seg:02d}"

    lats, lons = read_gps(seg_dir)
    if lats is None or len(lats) < 20:
        return None

    x, y = latlon_to_xy(lats, lons)

    # Total trajectory length from raw GPS
    dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    traj_len = float(dists.sum())
    if traj_len < 5.0:
        return None

    # Resample at DIST_STEP_M intervals to suppress GPS noise
    rx, ry = resample_by_distance(x, y, DIST_STEP_M)
    if len(rx) < 4:
        return None

    headings   = compute_headings(rx, ry)
    changes    = compute_heading_changes(headings, SMOOTH_WINDOW)
    events     = detect_turn_events(changes, TURN_EVENT_THRESH)

    total_heading = float(np.abs(changes).sum())
    net_heading   = float(changes.sum())
    max_L = max((abs(e["total_deg"]) for e in events if e["direction"] == "L"), default=0.0)
    max_R = max((abs(e["total_deg"]) for e in events if e["direction"] == "R"), default=0.0)
    n_L   = sum(1 for e in events if e["direction"] == "L")
    n_R   = sum(1 for e in events if e["direction"] == "R")

    stats = {
        "episode":             ep,
        "segment":             seg,
        "seg_key":             f"episode_{ep:04d}_seg{seg:02d}",
        "n_frames":            len(lats),
        "n_waypoints":         len(rx),  # after 2m resampling
        "traj_len_m":          traj_len,
        "total_heading_deg":   total_heading,
        "net_heading_deg":     net_heading,
        "max_left_turn_deg":   max_L,
        "max_right_turn_deg":  max_R,
        "n_turn_events":       len(events),
        "n_left_events":       n_L,
        "n_right_events":      n_R,
        "curvature_density":   total_heading / traj_len * 100,  # deg / 100m
        "turn_events":         events,
        # from episode_scores.json
        "heading_err_deg":     seg_info.get("heading_err_deg"),
        "frechet_norm":        seg_info.get("frechet_norm"),
    }
    stats["scenario"] = classify_segment(stats)
    return stats


# ── Visualization ─────────────────────────────────────────────────────────────

SCENARIO_COLORS = {
    "straight":         "#4CAF50",
    "curve_left":       "#2196F3",
    "curve_right":      "#03A9F4",
    "sharp_left_turn":  "#9C27B0",
    "sharp_right_turn": "#E91E63",
    "intersection":     "#FF9800",
    "winding":          "#FF5722",
    "complex_urban":    "#795548",
}


def plot_pie(counts, out_path):
    labels = list(counts.keys())
    sizes  = [counts[l] for l in labels]
    colors = [SCENARIO_COLORS.get(l, "#888") for l in labels]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.82,
        wedgeprops=dict(edgecolor='white', linewidth=1.5)
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title("Dataset Scenario Distribution (n=538 segments)", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_curvature_histogram(all_stats, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Total heading change
    vals = [s["total_heading_deg"] for s in all_stats]
    axes[0].hist(vals, bins=40, color="#2196F3", edgecolor='white')
    axes[0].set_xlabel("Total heading change (deg)")
    axes[0].set_ylabel("# segments")
    axes[0].set_title("Total Trajectory Curvature")

    # 2. Net heading change (signed)
    vals = [s["net_heading_deg"] for s in all_stats]
    axes[1].hist(vals, bins=40, color="#4CAF50", edgecolor='white')
    axes[1].axvline(0, color='red', lw=1, ls='--')
    axes[1].set_xlabel("Net heading change (deg, + = left)")
    axes[1].set_title("Net Heading Change (Direction Bias)")

    # 3. Curvature density
    vals = [s["curvature_density"] for s in all_stats]
    axes[2].hist(vals, bins=40, color="#FF9800", edgecolor='white')
    axes[2].axvline(CURV_DENSITY_THRESH, color='red', lw=1, ls='--', label=f'urban thresh ({CURV_DENSITY_THRESH}°/100m)')
    axes[2].legend(fontsize=8)
    axes[2].set_xlabel("Curvature density (deg / 100m)")
    axes[2].set_title("Curvature Density")

    fig.suptitle("Trajectory Curvature Statistics (n=538 segments)", fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_scatter(all_stats, out_path):
    """Scatter: total_heading vs net_heading, colored by scenario."""
    fig, ax = plt.subplots(figsize=(9, 7))

    scenario_groups = defaultdict(list)
    for s in all_stats:
        scenario_groups[s["scenario"]].append(s)

    for scenario, group in sorted(scenario_groups.items()):
        xs = [s["net_heading_deg"]   for s in group]
        ys = [s["total_heading_deg"] for s in group]
        ax.scatter(xs, ys, label=f"{scenario} (n={len(group)})",
                   color=SCENARIO_COLORS.get(scenario, "#888"),
                   alpha=0.7, s=30, edgecolors='none')

    ax.axvline(0, color='gray', lw=0.5, ls='--')
    ax.set_xlabel("Net heading change (deg, + = left, - = right)")
    ax.set_ylabel("Total heading change (deg, unsigned)")
    ax.set_title("Scenario Classification Space")
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_traj_len_distribution(all_stats, out_path):
    """Bar chart: scenario × trajectory length buckets."""
    buckets = ["30-50m", "50-100m", "100-200m", "200-425m"]
    def bucket(l):
        if l < 50:   return "30-50m"
        if l < 100:  return "50-100m"
        if l < 200:  return "100-200m"
        return "200-425m"

    scenarios = sorted(SCENARIO_COLORS.keys())
    data = {sc: Counter() for sc in scenarios}
    for s in all_stats:
        data[s["scenario"]][bucket(s["traj_len_m"])] += 1

    x = np.arange(len(buckets))
    width = 0.8 / len(scenarios)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, sc in enumerate(scenarios):
        counts = [data[sc].get(b, 0) for b in buckets]
        ax.bar(x + i * width, counts, width, label=sc,
               color=SCENARIO_COLORS.get(sc, "#888"), alpha=0.85)

    ax.set_xticks(x + width * len(scenarios) / 2)
    ax.set_xticklabels(buckets)
    ax.set_ylabel("# segments")
    ax.set_title("Scenario Type × Trajectory Length")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_trajectory_grid(all_stats, out_path, n_per_scenario=6):
    """Mini trajectory plots: grid of examples per scenario."""
    scenarios = sorted(set(s["scenario"] for s in all_stats))
    n_cols = n_per_scenario
    n_rows = len(scenarios)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    by_scenario = defaultdict(list)
    for s in all_stats:
        by_scenario[s["scenario"]].append(s)

    for row, scenario in enumerate(scenarios):
        samples = by_scenario[scenario][:n_cols]
        for col in range(n_cols):
            ax = axes[row, col]
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(scenario, fontsize=7, rotation=90, labelpad=2)
            if col < len(samples):
                s = samples[col]
                seg_dir = OSM_ROOT / s["seg_key"]
                lats, lons = read_gps(seg_dir)
                if lats is not None:
                    x, y = latlon_to_xy(lats, lons)
                    ax.plot(x, y, lw=1.0,
                            color=SCENARIO_COLORS.get(scenario, "#888"))
                    ax.plot(x[0], y[0], 'go', ms=3)
                    ax.plot(x[-1], y[-1], 'rs', ms=3)
                    ax.set_aspect('equal')
                    ax.set_title(f"{s['seg_key'][-8:]}\n{s['traj_len_m']:.0f}m",
                                 fontsize=5)

    fig.suptitle("Sample Trajectories by Scenario Type  (green=start, red=end)",
                 fontsize=10, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_map_scale_analysis(all_stats, out_path):
    """
    Analyzes appropriate map observation radius for navigation.
    For each segment, computes:
      - 'lookahead_m': distance from current position to the next turn event
      - This tells us how far ahead we need to see on the map to anticipate maneuvers
    """
    lookaheads = []
    for s in all_stats:
        events = s.get("turn_events", [])
        if events and s.get("n_waypoints", 0) > 0:
            # start is a waypoint index in the 2m-resampled array → multiply by DIST_STEP_M
            lookaheads.append(events[0]["start"] * DIST_STEP_M)

    if not lookaheads:
        return

    lookaheads = np.array(lookaheads)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lookaheads, bins=40, color="#9C27B0", edgecolor='white', alpha=0.8)

    pct25 = np.percentile(lookaheads, 25)
    pct50 = np.percentile(lookaheads, 50)
    pct75 = np.percentile(lookaheads, 75)
    ax.axvline(pct25, color='blue',   lw=1.5, ls='--', label=f'25th pct: {pct25:.1f}m')
    ax.axvline(pct50, color='green',  lw=1.5, ls='-',  label=f'50th pct: {pct50:.1f}m')
    ax.axvline(pct75, color='orange', lw=1.5, ls='--', label=f'75th pct: {pct75:.1f}m')

    ax.set_xlabel("Distance to first turn event (m)")
    ax.set_ylabel("# segments")
    ax.set_title("Map Lookahead Distance Analysis\n(how far ahead model needs to see to anticipate turns)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    print(f"\n  Map scale recommendation:")
    print(f"    25th pct lookahead: {pct25:.1f}m")
    print(f"    Median  lookahead:  {pct50:.1f}m")
    print(f"    75th pct lookahead: {pct75:.1f}m")
    print(f"    → Recommend map radius of at least {max(pct75, 20):.0f}m to cover 75% of turns")
    print(f"    Current OSM map: MAP_RANGE_M=25m (50x50m), 224px → {50/224:.2f}m/px")
    print(f"    At 96px (model input): {50/96:.2f}m/px — sufficient for campus navigation")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Dataset Distribution Analysis — rides_11 / OSM Maps Arrow")
    print("=" * 60)

    selected = load_selected_segments()
    print(f"\nLoaded {len(selected)} selected segments from episode_scores.json")

    print("\nAnalyzing trajectories...")
    all_stats = []
    skipped = 0
    for i, seg_info in enumerate(selected):
        stats = analyze_segment(seg_info)
        if stats is None:
            skipped += 1
            continue
        all_stats.append(stats)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(selected)} done...")

    print(f"\nAnalyzed: {len(all_stats)} segments ({skipped} skipped — GPS missing/short)")

    # ── Summary ──────────────────────────────────────────────────────────────
    scenario_counts = Counter(s["scenario"] for s in all_stats)
    total = len(all_stats)

    print("\n" + "─" * 50)
    print("SCENARIO DISTRIBUTION")
    print("─" * 50)
    for sc, cnt in sorted(scenario_counts.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {sc:20s}  {cnt:4d}  ({pct:5.1f}%)  {bar}")

    print("\nKEY STATS")
    print("─" * 50)
    traj_lens = [s["traj_len_m"] for s in all_stats]
    total_heads = [s["total_heading_deg"] for s in all_stats]
    curv_dens   = [s["curvature_density"] for s in all_stats]
    print(f"  Trajectory length (m):    "
          f"min={min(traj_lens):.1f}  median={np.median(traj_lens):.1f}  "
          f"max={max(traj_lens):.1f}")
    print(f"  Total heading change (deg): "
          f"min={min(total_heads):.1f}  median={np.median(total_heads):.1f}  "
          f"max={max(total_heads):.1f}")
    print(f"  Curvature density (deg/100m): "
          f"min={min(curv_dens):.1f}  median={np.median(curv_dens):.1f}  "
          f"max={max(curv_dens):.1f}")

    # ── Difficulty tags ───────────────────────────────────────────────────────
    def difficulty(s):
        sc = s["scenario"]
        if sc == "straight":
            return "easy"
        if sc in ("curve_left", "curve_right"):
            return "medium"
        if sc in ("sharp_left_turn", "sharp_right_turn"):
            return "hard"
        if sc in ("intersection", "winding"):
            return "hard"
        if sc == "complex_urban":
            return "very_hard"
        return "medium"

    for s in all_stats:
        s["difficulty"] = difficulty(s)
        # Tag list for filtering
        tags = [s["scenario"]]
        if s["n_left_events"] > 0:   tags.append("left_turn")
        if s["n_right_events"] > 0:  tags.append("right_turn")
        if s["n_turn_events"] >= 2:  tags.append("multi_turn")
        if s["curvature_density"] > CURV_DENSITY_THRESH: tags.append("high_curvature")
        if s["traj_len_m"] > 150:    tags.append("long")
        if s["traj_len_m"] < 40:     tags.append("short")
        s["tags"] = tags

    diff_counts = Counter(s["difficulty"] for s in all_stats)
    print("\nDIFFICULTY DISTRIBUTION")
    print("─" * 50)
    for d, cnt in [("easy","easy"),("medium","medium"),("hard","hard"),("very_hard","very_hard")]:
        cnt = diff_counts.get(d, 0)
        print(f"  {d:10s}  {cnt:4d}  ({cnt/total*100:5.1f}%)")

    # ── Save labels ───────────────────────────────────────────────────────────
    labels_path = OUT_DIR / "dataset_labels.json"
    # Remove turn_events list (verbose) from saved JSON — keep summary
    save_stats = []
    for s in all_stats:
        ss = {k: v for k, v in s.items() if k != "turn_events"}
        save_stats.append(ss)

    with open(labels_path, "w") as f:
        json.dump(save_stats, f, indent=2)
    print(f"\nSaved labels → {labels_path}")

    # ── Summary CSV ───────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "dataset_labels.csv"
    fields = ["seg_key","episode","segment","scenario","difficulty","tags",
              "n_frames","traj_len_m","total_heading_deg","net_heading_deg",
              "max_left_turn_deg","max_right_turn_deg","n_turn_events",
              "n_left_events","n_right_events","curvature_density",
              "heading_err_deg","frechet_norm"]
    with open(csv_path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for s in save_stats:
            row = dict(s)
            row["tags"] = "|".join(s.get("tags", []))
            writer.writerow(row)
    print(f"Saved CSV    → {csv_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating visualizations...")
    plot_pie(scenario_counts,       OUT_DIR / "01_scenario_distribution.png")
    plot_curvature_histogram(all_stats, OUT_DIR / "02_curvature_histograms.png")
    plot_scatter(all_stats,         OUT_DIR / "03_classification_scatter.png")
    plot_traj_len_distribution(all_stats, OUT_DIR / "04_scenario_by_length.png")
    plot_trajectory_grid(all_stats, OUT_DIR / "05_trajectory_examples.png", n_per_scenario=6)
    plot_map_scale_analysis(all_stats, OUT_DIR / "06_map_scale_analysis.png")

    # ── Per-frame distribution ─────────────────────────────────────────────────
    from collections import defaultdict as _dd
    frame_counts = _dd(int)
    for s in all_stats:
        frame_counts[s["scenario"]] += s["n_frames"]
    total_frames = sum(frame_counts.values())

    print("\nPER-FRAME DISTRIBUTION (actual training sample weight)")
    print("─" * 50)
    for sc, cnt in sorted(frame_counts.items(), key=lambda x: -x[1]):
        pct = cnt / total_frames * 100
        bar = "█" * int(pct / 2)
        print(f"  {sc:20s}  {cnt:7d}  ({pct:5.1f}%)  {bar}")
    print(f"  {'TOTAL':20s}  {total_frames:7d}")

    straight_pct = frame_counts.get("straight", 0) / total_frames * 100
    print(f"\n  Straight-road frames: {straight_pct:.1f}%")
    print(f"  Non-straight frames: {100-straight_pct:.1f}%")
    if straight_pct < 20:
        print("  → Dataset is NOT dominated by straight driving.")
    else:
        print("  ⚠ Dataset may be biased toward straight driving — consider rebalancing.")

    print("\nAll outputs saved to:", OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
