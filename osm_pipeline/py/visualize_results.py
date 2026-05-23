"""
Visualization helper: inspect episode scores and OSM map samples.

Usage:
  python visualize_results.py --mode scores          # bar chart of metrics
  python visualize_results.py --mode maps --ep 6     # show OSM maps for ep 6
  python visualize_results.py --mode overlay --ep 6  # EKF vs OSM path overlay
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


SCORES_PATH = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/episode_scores.json"
MAPS_ROOT   = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/osm_maps"
ARROW_PATH  = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/FrodoBots-2K/processed/frodobots_dataset/train/data-00000-of-00001.arrow"


def plot_scores():
    with open(SCORES_PATH) as f:
        scores = json.load(f)

    scored = [s for s in scores if 'frechet_norm' in s]
    if not scored:
        print("No scored episodes found yet.")
        return

    eps        = [s['episode'] for s in scored]
    frechet    = [s['frechet_norm'] for s in scored]
    chamfer    = [s['chamfer_norm'] for s in scored]
    heading    = [s['heading_err_deg'] for s in scored]
    selected   = [s['selected'] for s in scored]
    colors     = ['green' if s else 'red' for s in selected]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].bar(eps, frechet, color=colors, alpha=0.7)
    axes[0].axhline(0.5, color='k', linestyle='--', linewidth=1)
    axes[0].set_ylabel("Fréchet (norm)")

    axes[1].bar(eps, chamfer, color=colors, alpha=0.7)
    axes[1].axhline(0.3, color='k', linestyle='--', linewidth=1)
    axes[1].set_ylabel("Chamfer (norm)")

    axes[2].bar(eps, heading, color=colors, alpha=0.7)
    axes[2].axhline(30, color='k', linestyle='--', linewidth=1)
    axes[2].set_ylabel("Heading err (°)")
    axes[2].set_xlabel("Episode")

    n_sel = sum(selected)
    fig.suptitle(f"Episode scores  —  {n_sel}/{len(scored)} selected")
    green_patch = mpatches.Patch(color='green', label='Selected')
    red_patch   = mpatches.Patch(color='red', label='Rejected')
    fig.legend(handles=[green_patch, red_patch], loc='upper right')

    out = os.path.join(os.path.dirname(SCORES_PATH), "episode_scores.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"Saved: {out}")
    plt.show()


def plot_maps(ep):
    ep_dir = os.path.join(MAPS_ROOT, f"episode_{ep:04d}")
    if not os.path.exists(ep_dir):
        print(f"No maps for episode {ep}")
        return

    from PIL import Image
    import glob
    maps = sorted(glob.glob(os.path.join(ep_dir, "osm_map_*.png")))
    if not maps:
        print("No map images found.")
        return

    n = min(12, len(maps))
    idxs = np.linspace(0, len(maps)-1, n, dtype=int)
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    for ax, idx in zip(axes.flat, idxs):
        img = np.array(Image.open(maps[idx]))
        ax.imshow(img)
        ax.set_title(f"frame {idx}", fontsize=8)
        ax.axis('off')
    fig.suptitle(f"Episode {ep} — OSM local path maps")
    out = os.path.join(ep_dir, "sample_maps.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"Saved: {out}")
    plt.show()


def plot_overlay(ep):
    """Overlay EKF trajectory vs OSM-matched path."""
    import pyarrow as pa
    import osmnx as ox
    import networkx as nx
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from episode_selector import fetch_osm_graph, map_match_trajectory, latlon_to_utm_xy

    table = pa.ipc.open_stream(open(ARROW_PATH, 'rb')).read_all()
    ep_idx = np.array(table['episode_index'].to_pylist())
    lats   = np.array(table['observation.latitude'].to_pylist())
    lons   = np.array(table['observation.longitude'].to_pylist())
    fp     = np.array(table['observation.filtered_position'].to_pylist())

    mask = ep_idx == ep
    ep_lats = lats[mask]
    ep_lons = lons[mask]
    ep_fp   = fp[mask]

    print(f"Fetching OSM graph for episode {ep}...")
    G = fetch_osm_graph(ep_lats, ep_lons)
    if G is None:
        print("OSM fetch failed.")
        return

    print("Map-matching...")
    osm_path = map_match_trajectory(G, ep_lats, ep_lons)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(ep_fp[:, 0], ep_fp[:, 1], 'b-o', markersize=2, label='EKF trajectory')
    if osm_path is not None:
        ax.plot(osm_path[:, 0], osm_path[:, 1], 'r-', linewidth=2, label='OSM matched path')
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title(f"Episode {ep}: EKF vs OSM path")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")

    out = os.path.join(os.path.dirname(SCORES_PATH), f"overlay_ep{ep:04d}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"Saved: {out}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scores", "maps", "overlay"], default="scores")
    parser.add_argument("--ep", type=int, default=6)
    args = parser.parse_args()

    if args.mode == "scores":
        plot_scores()
    elif args.mode == "maps":
        plot_maps(args.ep)
    elif args.mode == "overlay":
        plot_overlay(args.ep)


if __name__ == "__main__":
    main()
