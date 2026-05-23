"""
build_global_odom_map.py

Generates a per-episode global odometry map image using raw GPS pos/yaw
from gnm_format traj_data.pkl.

Each map shows:
  - Full GPS trajectory of the episode (white line)
  - Frame positions as dots, colour-coded by speed
  - Start marker (green) and end marker (red)
  - Heading arrows every N frames
  - Scale bar and metadata text

Output:
  <out_root>/<episode_id>_global_odom.png

Usage:
  python3 build_global_odom_map.py \
      --gnm_root /path/to/FrodoBots-2K/gnm_format \
      --out_root /path/to/output_dir \
      [--img_px 800] [--margin 0.1] [--workers 8]
"""

import argparse
import os
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm


# ── helpers ───────────────────────────────────────────────────────────────────

def _render_global_map(pos: np.ndarray,
                       yaw: np.ndarray,
                       episode_id: str,
                       img_px: int = 800,
                       margin: float = 0.12) -> np.ndarray:
    """
    Render a single episode's GPS trajectory as a top-down map image.

    Coordinate layout (standard map convention):
        +X → rightward  (East)
        +Y → upward     (North)
        Image col = px_x,  Image row = (H-1) - px_y   (flip Y for image)

    Args:
        pos        : (F, 2) GPS positions in metres (episode-relative origin)
        yaw        : (F,)   headings in radians
        episode_id : string for title
        img_px     : output image height & width in pixels
        margin     : fractional padding around trajectory (0.0–0.5)

    Returns:
        uint8 BGR image (img_px, img_px, 3)
    """
    F = len(pos)
    if F < 2:
        canvas = np.zeros((img_px, img_px, 3), np.uint8)
        cv2.putText(canvas, 'too short', (10, img_px // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
        return canvas

    x = pos[:, 0].copy()
    y = pos[:, 1].copy()

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_span = max(x_max - x_min, 1e-3)
    y_span = max(y_max - y_min, 1e-3)

    # Keep aspect ratio: fit the longer axis to img_px * (1 - 2*margin)
    draw_px = int(img_px * (1 - 2 * margin))
    scale = draw_px / max(x_span, y_span)   # px per metre

    # Offset so trajectory is centred
    x_centre = (x_min + x_max) / 2
    y_centre = (y_min + y_max) / 2
    cx = img_px / 2 + (x - x_centre) * scale
    # Flip Y: GPS +Y = North = up in map, but image row increases downward
    cy = img_px / 2 - (y - y_centre) * scale

    cx = cx.astype(np.int32)
    cy = cy.astype(np.int32)

    canvas = np.zeros((img_px, img_px, 3), np.uint8)

    # ── speed-coded dots ─────────────────────────────────────────────────────
    # speed = displacement between consecutive frames (m per frame)
    speeds = np.concatenate([[0.0],
                              np.linalg.norm(np.diff(pos, axis=0), axis=1)])
    v_max = np.percentile(speeds, 95) + 1e-6

    for i in range(F):
        v_norm = min(speeds[i] / v_max, 1.0)
        # colour: blue (slow) → cyan → green → yellow → red (fast)
        hue = int((1.0 - v_norm) * 120)          # 120=green(slow) → 0=red(fast)
        bgr = cv2.cvtColor(
            np.array([[[hue, 220, 200]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
        dot_r = max(2, img_px // 200)
        cv2.circle(canvas, (cx[i], cy[i]), dot_r, bgr.tolist(), -1)

    # ── trajectory polyline ───────────────────────────────────────────────────
    pts = np.stack([cx, cy], axis=1).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], isClosed=False,
                  color=(200, 200, 200), thickness=1, lineType=cv2.LINE_AA)

    # ── heading arrows every ~5% of frames ───────────────────────────────────
    arrow_interval = max(1, F // 20)
    arrow_len = max(6, img_px // 80)
    for i in range(0, F, arrow_interval):
        ax = int(cx[i] + arrow_len * np.cos(yaw[i]))
        ay = int(cy[i] - arrow_len * np.sin(yaw[i]))   # flip Y
        cv2.arrowedLine(canvas, (cx[i], cy[i]), (ax, ay),
                        (0, 255, 255), 1, cv2.LINE_AA, tipLength=0.4)

    # ── start / end markers ───────────────────────────────────────────────────
    marker_r = max(6, img_px // 80)
    cv2.circle(canvas, (cx[0],  cy[0]),  marker_r, (0, 255, 0), -1)   # green start
    cv2.circle(canvas, (cx[-1], cy[-1]), marker_r, (0, 0, 255), -1)   # red end
    cv2.putText(canvas, 'S', (cx[0]  + marker_r + 2, cy[0]  + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'E', (cx[-1] + marker_r + 2, cy[-1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    # ── scale bar ────────────────────────────────────────────────────────────
    # Pick a round scale (1, 2, 5, 10, 20, 50 m) that is ~15% of img width
    target_m = img_px * 0.15 / scale
    for s in [1, 2, 5, 10, 20, 50, 100, 200]:
        if s >= target_m * 0.5:
            scale_m = s
            break
    else:
        scale_m = 200
    bar_px = int(scale_m * scale)
    bx0 = int(img_px * 0.05)
    bx1 = bx0 + bar_px
    by  = img_px - int(img_px * 0.06)
    cv2.line(canvas, (bx0, by), (bx1, by), (255, 255, 255), 2)
    cv2.line(canvas, (bx0, by - 4), (bx0, by + 4), (255, 255, 255), 2)
    cv2.line(canvas, (bx1, by - 4), (bx1, by + 4), (255, 255, 255), 2)
    cv2.putText(canvas, f'{scale_m} m',
                (bx0, by - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # ── metadata text ─────────────────────────────────────────────────────────
    path_len = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))
    info_lines = [
        episode_id[-26:],
        f'frames: {F}',
        f'path:   {path_len:.1f} m',
        f'extent: {x_span:.1f} x {y_span:.1f} m',
    ]
    for li, line in enumerate(info_lines):
        cv2.putText(canvas, line,
                    (int(img_px * 0.04), int(img_px * 0.04) + li * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (180, 180, 180), 1, cv2.LINE_AA)

    # ── colour legend ─────────────────────────────────────────────────────────
    lx = img_px - int(img_px * 0.22)
    ly = img_px - int(img_px * 0.10)
    for i, (label, hue) in enumerate([('slow', 120), ('fast', 0)]):
        bgr = cv2.cvtColor(
            np.array([[[hue, 220, 200]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
        cv2.circle(canvas, (lx, ly + i * 16), 5, bgr.tolist(), -1)
        cv2.putText(canvas, label, (lx + 10, ly + i * 16 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (180, 180, 180), 1, cv2.LINE_AA)

    return canvas


# ── worker ───────────────────────────────────────────────────────────────────

def _process_episode(args_tuple):
    ep_dir, out_root, img_px, margin = args_tuple
    ep_id    = Path(ep_dir).name
    pkl_path = Path(ep_dir) / 'traj_data.pkl'

    if not pkl_path.exists():
        return ep_id, False, 'no pkl'

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    pos = np.array(data['position'], dtype=np.float64)
    yaw = np.array(data['yaw'],      dtype=np.float64)

    if len(pos) < 2:
        return ep_id, False, 'too short'

    img = _render_global_map(pos, yaw, ep_id, img_px, margin)
    out_path = Path(out_root) / f'{ep_id}_global_odom.png'
    cv2.imwrite(str(out_path), img)
    return ep_id, True, None


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gnm_root', required=True)
    parser.add_argument('--out_root', required=True)
    parser.add_argument('--img_px',   type=int,   default=800)
    parser.add_argument('--margin',   type=float, default=0.12)
    parser.add_argument('--workers',  type=int,   default=8)
    args = parser.parse_args()

    gnm_root = Path(args.gnm_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    episodes = sorted([
        str(gnm_root / ep)
        for ep in os.listdir(gnm_root)
        if (gnm_root / ep / 'traj_data.pkl').exists()
    ])
    print(f'Found {len(episodes)} episodes  →  output: {out_root}')

    ok, fail = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_process_episode,
                        (ep, str(out_root), args.img_px, args.margin)): ep
            for ep in episodes
        }
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc='Episodes', unit='ep'):
            ep_id, success, reason = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                print(f'  SKIP {ep_id}: {reason}')

    print(f'\nDone.  saved={ok}  skipped={fail}  →  {out_root}')


if __name__ == '__main__':
    main()
