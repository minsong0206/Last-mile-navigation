"""
build_odom_dataset.py

Reads gnm_format episodes and writes a new dataset where each sample contains:
  - current image   (t)
  - past image -1   (t-1)
  - past image -2   (t-2)
  - odom map        (t)  ← 8-step local BEV rasterized from GPS pos/yaw

Output folder structure (mirrors gnm_format per episode):

  <output_root>/
    <episode_id>/
      samples.pkl          ← list of dicts, one per valid frame t
        {
          'frame_idx'   : int,          # t within episode
          'pos'         : (2,) float,   # GPS pos at t (metres, global)
          'yaw'         : float,        # GPS yaw at t (radians)
          'action_norm' : (8,4) float,  # normalised [fwd,left,cos,sin] from GPS
        }
      <t>/
        current.jpg        ← image at t
        past_1.jpg         ← image at t-1
        past_2.jpg         ← image at t-2
        odom_map.png       ← 8-step local BEV (64×64 px)

Usage:
  python3 build_odom_dataset.py \
      --gnm_root  /path/to/FrodoBots-2K/gnm_format \
      --out_root  /path/to/output_dataset \
      [--img_px 64] [--forward_m 1.0] [--lateral_m 0.5] [--workers 8]
"""

import argparse
import os
import pickle
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm

# ── shared constants ─────────────────────────────────────────────────────────
TRAIN_METRIC_SPACING = 0.25 * 0.5   # 0.125 m/unit (matches train_utils.py)
N_STEPS = 8
PAST_FRAMES = 2                      # t-1, t-2


# ── odom map rasterizer (same logic as trajectory_map_generator.py) ──────────

def _odom_map(action_norm_t: np.ndarray,
              img_px: int,
              forward_range_m: float,
              lateral_range_m: float) -> np.ndarray:
    """
    Rasterize one frame's 8-step local trajectory into a BEV image.

    Layout:
        robot origin : bottom-centre
        forward (+X) : upward  in image
        left    (+Y) : leftward in image

    Args:
        action_norm_t : (N_STEPS, 4) normalised [fwd, left, cos_yaw, sin_yaw]
    Returns:
        uint8 BGR image (img_px, img_px, 3)
    """
    xy_m = action_norm_t[:, :2] * TRAIN_METRIC_SPACING   # metres

    canvas = np.zeros((img_px, img_px, 3), dtype=np.uint8)

    robot_row = int(img_px * 0.92)
    robot_col = img_px // 2

    def m2px(fwd_m, left_m):
        col = int(robot_col - left_m  / lateral_range_m * (img_px / 2))
        row = int(robot_row - fwd_m   / forward_range_m * (img_px * 0.92))
        return col, row

    # Robot origin: red dot
    origin_r = max(4, img_px // 20)
    cv2.circle(canvas, (robot_col, robot_row), origin_r, (0, 0, 255), -1)

    pts = [(robot_col, robot_row)] + \
          [m2px(xy_m[i, 0], xy_m[i, 1]) for i in range(N_STEPS)]

    line_thick = max(2, img_px // 48)
    dot_r      = max(3, img_px // 32)
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(canvas, a, b, (255, 255, 255), line_thick, cv2.LINE_AA)
    for p in pts[1:]:
        cv2.circle(canvas, p, dot_r, (0, 255, 0), -1)

    return canvas


# ── per-frame action builder from GPS ────────────────────────────────────────

def _yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array([[np.cos(yaw), -np.sin(yaw)],
                     [np.sin(yaw),  np.cos(yaw)]])


def _gps_to_action_norm(pos: np.ndarray, yaw: np.ndarray,
                         t: int) -> np.ndarray:
    """
    Build action_estfrod-compatible (N_STEPS, 4) array for frame t using GPS.

    Heading convention in FrodoBots gnm_format:
        yaw[t] stores the heading that was computed AFTER arriving at frame t,
        i.e. it equals the movement direction of the step t-1 → t.
        Therefore yaw[t+1] is the actual robot heading AT frame t
        (the direction the robot is facing when it leaves frame t).

    Transform: local = (pos[t+k+1] - pos[t]) @ R(yaw[t+1])
        col 0 of R → forward  (+X in robot frame)
        col 1 of R → left     (+Y in robot frame)

    rel_yaw = yaw[t+k+1] - yaw[t+1],  normalised to [-pi, pi]
    """
    # yaw[t+1] is the robot heading at the moment of leaving frame t
    curr_yaw = yaw[t + 1]
    rotmat   = _yaw_rotmat(curr_yaw)
    curr_pos = pos[t]

    action = np.zeros((N_STEPS, 4), dtype=np.float32)
    for k in range(N_STEPS):
        delta    = pos[t + k + 1] - curr_pos          # (2,) global displacement
        local    = delta @ rotmat                      # (2,) [fwd, left]
        rel_yaw  = yaw[t + k + 1] - curr_yaw
        rel_yaw  = (rel_yaw + np.pi) % (2 * np.pi) - np.pi

        action[k, 0] = local[0] / TRAIN_METRIC_SPACING
        action[k, 1] = local[1] / TRAIN_METRIC_SPACING
        action[k, 2] = np.cos(rel_yaw)
        action[k, 3] = np.sin(rel_yaw)

    return action


# ── process one episode ───────────────────────────────────────────────────────

def process_episode(args_tuple):
    """
    Worker function (runs in subprocess via ProcessPoolExecutor).

    Returns (episode_id, n_samples, skip_reason_or_None)
    """
    ep_dir, out_root, img_px, forward_m, lateral_m = args_tuple

    ep_id   = Path(ep_dir).name
    pkl_path = Path(ep_dir) / 'traj_data.pkl'

    if not pkl_path.exists():
        return ep_id, 0, 'no traj_data.pkl'

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    pos = np.array(data['pos'], dtype=np.float64)   # (F, 2)
    yaw = np.array(data['yaw'], dtype=np.float64)   # (F,)
    F   = len(pos)

    # Need: t-2, t-1, t, t+1 … t+8  →  first valid t = PAST_FRAMES, last = F-N_STEPS-1
    first_t = PAST_FRAMES
    last_t  = F - N_STEPS - 1

    if last_t < first_t:
        return ep_id, 0, f'too short (F={F})'

    ep_out = Path(out_root) / ep_id
    ep_out.mkdir(parents=True, exist_ok=True)

    samples_meta = []

    for t in range(first_t, last_t + 1):
        sample_dir = ep_out / str(t)
        sample_dir.mkdir(exist_ok=True)

        # ── images ──────────────────────────────────────────────────────────
        for rel, name in [(0, 'current.jpg'), (-1, 'past_1.jpg'), (-2, 'past_2.jpg')]:
            src = Path(ep_dir) / f'{t + rel}.jpg'
            dst = sample_dir / name
            if not dst.exists():
                shutil.copy2(src, dst)

        # ── action from GPS ──────────────────────────────────────────────────
        action_norm = _gps_to_action_norm(pos, yaw, t)

        # ── odom map ─────────────────────────────────────────────────────────
        odom = _odom_map(action_norm, img_px, forward_m, lateral_m)
        cv2.imwrite(str(sample_dir / 'odom_map.png'), odom)

        # ── metadata entry ───────────────────────────────────────────────────
        samples_meta.append({
            'frame_idx'  : t,
            'pos'        : pos[t].astype(np.float32),
            'yaw'        : float(yaw[t]),
            'action_norm': action_norm,
        })

    # Save episode-level metadata
    with open(ep_out / 'samples.pkl', 'wb') as f:
        pickle.dump(samples_meta, f)

    return ep_id, len(samples_meta), None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gnm_root',   required=True,
                        help='Path to FrodoBots-2K/gnm_format')
    parser.add_argument('--out_root',   required=True,
                        help='Output dataset root directory')
    parser.add_argument('--img_px',     type=int,   default=64,
                        help='Odom map pixel size (square)')
    parser.add_argument('--forward_m',  type=float, default=0.5,
                        help='Forward visible range in metres for odom map '
                             '(p50 of 8-step fwd displacement ≈ 0.42 m)')
    parser.add_argument('--lateral_m',  type=float, default=0.3,
                        help='Lateral visible range in metres for odom map '
                             '(p90 of lateral displacement ≈ 0.77 m)')
    parser.add_argument('--workers',    type=int,   default=8,
                        help='Number of parallel worker processes')
    args = parser.parse_args()

    gnm_root = Path(args.gnm_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    episodes = sorted([
        str(gnm_root / ep)
        for ep in os.listdir(gnm_root)
        if (gnm_root / ep / 'traj_data.pkl').exists()
    ])
    print(f"Found {len(episodes)} valid episodes in {gnm_root}")

    job_args = [
        (ep, str(out_root), args.img_px, args.forward_m, args.lateral_m)
        for ep in episodes
    ]

    total_samples = 0
    skipped = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_episode, a): a[0] for a in job_args}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc='Episodes', unit='ep'):
            ep_id, n, reason = fut.result()
            if reason:
                skipped.append((ep_id, reason))
            else:
                total_samples += n

    # Summary
    print(f"\n{'='*55}")
    print(f"Done.")
    print(f"  Episodes processed : {len(episodes) - len(skipped)}")
    print(f"  Episodes skipped   : {len(skipped)}")
    print(f"  Total samples      : {total_samples}")
    print(f"  Output             : {out_root}")
    print(f"{'='*55}")

    if skipped:
        print("Skipped episodes:")
        for ep_id, reason in skipped:
            print(f"  {ep_id}: {reason}")

    # Write global index
    index = {
        'total_samples': total_samples,
        'episodes': len(episodes) - len(skipped),
        'img_px': args.img_px,
        'forward_range_m': args.forward_m,
        'lateral_range_m': args.lateral_m,
        'train_metric_spacing': TRAIN_METRIC_SPACING,
        'n_steps': N_STEPS,
        'past_frames': PAST_FRAMES,
    }
    with open(out_root / 'dataset_index.pkl', 'wb') as f:
        pickle.dump(index, f)
    print(f"Index saved → {out_root / 'dataset_index.pkl'}")


if __name__ == '__main__':
    main()
