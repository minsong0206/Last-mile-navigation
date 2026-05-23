"""
Trajectory map generator from MBRA action_estfrod outputs.

Two distinct purposes:
  1. Training/inference  → local_traj_image()   : 8-step egocentric raster image
  2. Debug/visualization → three plot functions : long future, current-to-goal, overlay
"""

import sys
sys.path.insert(0, '../train')

import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants (must match train_utils.py)
# ---------------------------------------------------------------------------
TRAIN_METRIC_SPACING = 0.25 * 0.5   # 0.125 m/unit  — used during training
DT = 0.333                           # seconds per MBRA step
N_STEPS = 8                          # waypoints per action_estfrod frame


# ===========================================================================
# 1.  TRAINING / INFERENCE  —  Local egocentric trajectory image
# ===========================================================================

def local_traj_image(
    action_estfrod_t: np.ndarray,
    img_px: int = 64,
    forward_range_m: float = 2.0,
    lateral_range_m: float = 1.0,
    line_color: Tuple[int, int, int] = (255, 255, 255),
    point_color: Tuple[int, int, int] = (0, 255, 0),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Rasterize the 8-step local egocentric trajectory into a top-down BEV image.

    ── Coordinate convention (action_estfrod layout) ──────────────────────────
        action[..., 0]  = forward  displacement  (+X in robot frame)
                          normalised: divide by TRAIN_METRIC_SPACING (0.125 m)
                          positive → robot moves forward
        action[..., 1]  = left     displacement  (+Y in robot frame)
                          normalised: divide by TRAIN_METRIC_SPACING (0.125 m)
                          positive → robot turns/moves left
        action[..., 2]  = cos(yaw)   yaw accumulated from current heading
        action[..., 3]  = sin(yaw)   positive sin → left turn (CCW)

    ── Image layout ───────────────────────────────────────────────────────────
        Robot origin  : bottom-centre  (row = img_px - margin, col = img_px/2)
        Forward (+X)  : upward   in image  (row decreases)
        Left    (+Y)  : leftward in image  (col decreases)
        Right   (-Y)  : rightward in image (col increases)

    Args:
        action_estfrod_t  : np.ndarray (8, 4) — one batch element of action_estfrod.
        img_px            : output image side length in pixels (square).
        forward_range_m   : metric range visible above the robot (metres).
        lateral_range_m   : metric range visible left/right of the robot (metres).
        line_color        : BGR colour for trajectory polyline.
        point_color       : BGR colour for waypoint dots.
        bg_color          : BGR background colour.

    Returns:
        np.ndarray uint8 (img_px, img_px, 3) — egocentric BEV map.
        Robot is at bottom-centre; trajectory extends upward toward the future.
    """
    # Recover metric waypoints: normalised → metres
    xy_m = action_estfrod_t[:, :2] * TRAIN_METRIC_SPACING   # (8, 2)  [fwd_m, left_m]

    canvas = np.full((img_px, img_px, 3), bg_color, dtype=np.uint8)

    # Robot sits at bottom-centre with a small bottom margin (5 % of height)
    robot_row = int(img_px * 0.92)
    robot_col = img_px // 2

    def m2px(fwd_m: float, left_m: float) -> Tuple[int, int]:
        """
        Convert robot-frame metric coords to image pixel (col, row).
            fwd_m  > 0  → row decreases (upward in image)
            left_m > 0  → col decreases (leftward in image)
        """
        col = int(robot_col  - left_m    / lateral_range_m * (img_px / 2))
        row = int(robot_row  - fwd_m     / forward_range_m * (img_px * 0.92))
        return col, row

    # Robot origin marker (red dot at bottom-centre)
    cv2.circle(canvas, (robot_col, robot_row), max(2, img_px // 32), (0, 0, 255), -1)

    # Trajectory: origin → 8 waypoints
    pts = [(robot_col, robot_row)] + [m2px(xy_m[i, 0], xy_m[i, 1]) for i in range(N_STEPS)]
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(canvas, a, b, line_color, 1, cv2.LINE_AA)
    for p in pts[1:]:
        cv2.circle(canvas, p, max(1, img_px // 48), point_color, -1)

    return canvas


# ===========================================================================
# 2.  DEBUG / VISUALIZATION  —  Long future path
# ===========================================================================

def viz_long_future(
    action_estfrod_all: np.ndarray,
    frame_indices: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 8),
) -> plt.Figure:
    """
    [DEBUG ONLY — PSEUDO TRAJECTORY]

    Accumulate the *first waypoint* of each frame's MBRA local prediction via
    dead-reckoning to produce a long-range trajectory for sanity checking.

    !! This is NOT a ground-truth odometry map !!
    Sources of error:
      - Each frame's step-0 displacement is an independent MBRA prediction,
        not a direct measurement of actual robot motion.
      - Dead-reckoning accumulates heading errors: small per-frame yaw noise
        compounds over many frames, causing drift.
      - Use this only to verify that the trajectory shape is plausible and
        flows in the expected direction.  Do NOT use as training supervision.

    ── Accumulation convention ─────────────────────────────────────────────
        Per frame t, first waypoint gives (dx_norm, dy_norm, cos_yaw, sin_yaw).
        fwd_m  = dx_norm * 0.125   (metres, robot +X / forward)
        left_m = dy_norm * 0.125   (metres, robot +Y / left)
        dtheta = arctan2(sin_yaw, cos_yaw)   positive → left turn (CCW)

        global_x[t+1] = global_x[t] + cos(theta) * fwd_m - sin(theta) * left_m
        global_y[t+1] = global_y[t] + sin(theta) * fwd_m + cos(theta) * left_m
        theta[t+1]    = theta[t] + dtheta

    Args:
        action_estfrod_all : np.ndarray (T, 8, 4) — full episode actions.
        frame_indices      : subset of frame indices to accumulate (default: all).
        save_path          : if given, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    T = action_estfrod_all.shape[0]
    frames = frame_indices if frame_indices is not None else list(range(T))

    gx, gy, gtheta = 0.0, 0.0, 0.0
    global_xs = [gx]
    global_ys = [gy]

    for t in frames:
        dx_n  = action_estfrod_all[t, 0, 0]   # normalised forward  (+X)
        dy_n  = action_estfrod_all[t, 0, 1]   # normalised left     (+Y)
        cos_h = action_estfrod_all[t, 0, 2]
        sin_h = action_estfrod_all[t, 0, 3]
        # positive dtheta → left turn (CCW), consistent with robot +Y = left
        dtheta = np.arctan2(sin_h, cos_h)

        fwd_m  = dx_n * TRAIN_METRIC_SPACING
        left_m = dy_n * TRAIN_METRIC_SPACING

        # Rotate local displacement into global frame
        gx     += np.cos(gtheta) * fwd_m - np.sin(gtheta) * left_m
        gy     += np.sin(gtheta) * fwd_m + np.cos(gtheta) * left_m
        gtheta += dtheta

        global_xs.append(gx)
        global_ys.append(gy)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(global_xs, global_ys, '-o', markersize=2, linewidth=0.8,
            color='steelblue', label='pseudo-odom (dead-reckoning)')
    ax.plot(global_xs[0],  global_ys[0],  'go', markersize=8, label='start')
    ax.plot(global_xs[-1], global_ys[-1], 'rs', markersize=8, label='end')

    # Prominent pseudo-trajectory warning
    ax.text(0.01, 0.99,
            '⚠ PSEUDO TRAJECTORY\nDead-reckoning from MBRA predictions.\nNOT ground-truth odometry.',
            transform=ax.transAxes, fontsize=8, color='red',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    ax.set_aspect('equal')
    ax.set_xlabel('Global X — forward at t=0 (m)')
    ax.set_ylabel('Global Y — left at t=0 (m)')
    ax.set_title(f'[PSEUDO] Long future path  —  {len(frames)} frames\n'
                 f'(dead-reckoning, 1st waypoint/frame, spacing={TRAIN_METRIC_SPACING} m/unit)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ===========================================================================
# 3.  DEBUG / VISUALIZATION  —  Current-to-goal trajectory
# ===========================================================================

def viz_current_to_goal(
    action_estfrod_all: np.ndarray,
    start_frame: int,
    goal_frame: int,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 8),
) -> plt.Figure:
    """
    [DEBUG ONLY — PSEUDO TRAJECTORY]

    Dead-reckon the MBRA first-waypoint from start_frame to goal_frame and
    mark the endpoint as the "goal position estimate".

    ── What the goal marker means ───────────────────────────────────────────
        The red star (★) marks the *estimated* position of goal_frame relative
        to start_frame, computed by accumulating MBRA step-0 predictions.

        This is NOT the GPS ground-truth position of goal_frame.
        It represents: "if MBRA predictions were perfectly accurate and there
        were no drift, this is where the robot would be at goal_frame."

        Use it to check:
          • Does the path end up roughly in front of the robot? (sanity)
          • Is the trajectory smooth / free of sudden direction reversals?
          • Is the estimated goal distance plausible for the frame gap?

    ── Coordinate convention ────────────────────────────────────────────────
        Origin (0,0)  = robot position at start_frame
        +X axis       = robot forward direction at start_frame
        +Y axis       = robot left direction at start_frame
        Positive dtheta (sin_yaw > 0) → left turn (CCW)

    Args:
        action_estfrod_all : np.ndarray (T, 8, 4).
        start_frame        : current frame index; becomes the coordinate origin.
        goal_frame         : target frame index; its estimated position is marked.
        save_path          : optional output path.

    Returns:
        matplotlib Figure.
    """
    assert start_frame < goal_frame <= action_estfrod_all.shape[0], \
        "goal_frame must be > start_frame and within episode length"

    gx, gy, gtheta = 0.0, 0.0, 0.0
    xs = [gx]
    ys = [gy]

    for t in range(start_frame, goal_frame):
        dx_n  = action_estfrod_all[t, 0, 0]   # normalised forward  (+X)
        dy_n  = action_estfrod_all[t, 0, 1]   # normalised left     (+Y)
        cos_h = action_estfrod_all[t, 0, 2]
        sin_h = action_estfrod_all[t, 0, 3]
        dtheta = np.arctan2(sin_h, cos_h)     # positive → CCW / left turn

        fwd_m  = dx_n * TRAIN_METRIC_SPACING
        left_m = dy_n * TRAIN_METRIC_SPACING

        gx     += np.cos(gtheta) * fwd_m - np.sin(gtheta) * left_m
        gy     += np.sin(gtheta) * fwd_m + np.cos(gtheta) * left_m
        gtheta += dtheta

        xs.append(gx)
        ys.append(gy)

    dist = np.sqrt(xs[-1]**2 + ys[-1]**2)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(xs, ys, '-o', markersize=3, linewidth=1.0,
            color='darkorange', label='pseudo-odom path (dead-reckoning)')
    ax.plot(xs[0], ys[0], 'go', markersize=10, zorder=5,
            label=f'start  frame {start_frame}  (origin)')

    # Goal marker — with explicit explanation of what it represents
    ax.plot(xs[-1], ys[-1], 'r*', markersize=16, zorder=5,
            label=f'estimated goal position  frame {goal_frame}\n'
                  f'(NOT GPS ground-truth; MBRA dead-reckoning estimate)')
    ax.annotate(
        f'★ goal frame {goal_frame}\n'
        f'est. pos: ({xs[-1]:.2f} m fwd, {ys[-1]:.2f} m left)\n'
        f'est. dist from start: {dist:.2f} m',
        xy=(xs[-1], ys[-1]), xytext=(12, -30),
        textcoords='offset points', fontsize=8, color='darkred',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='mistyrose', alpha=0.8),
        arrowprops=dict(arrowstyle='->', color='darkred', lw=0.8),
    )

    # Reference circle: straight-line distance from origin to goal estimate
    circle = plt.Circle((0, 0), dist, fill=False, color='gray',
                         linestyle='--', linewidth=0.8, alpha=0.5)
    ax.add_patch(circle)
    ax.text(dist * 0.65, 0.02, f'straight-line {dist:.2f} m',
            color='gray', fontsize=7)

    # Forward axis arrow
    ax.annotate('', xy=(0.3, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    ax.text(0.32, 0.0, '+X fwd', fontsize=7, color='green')

    ax.text(0.01, 0.99,
            '⚠ PSEUDO TRAJECTORY — debug only\n'
            'Goal ★ = MBRA dead-reckoning estimate, NOT GPS truth.',
            transform=ax.transAxes, fontsize=8, color='red',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    ax.set_aspect('equal')
    ax.set_xlabel('X — forward at start_frame (m)')
    ax.set_ylabel('Y — left at start_frame (m)')
    ax.set_title(f'[PSEUDO] Current → Goal  '
                 f'(frames {start_frame} → {goal_frame},  Δ={goal_frame - start_frame} frames)\n'
                 f'spacing={TRAIN_METRIC_SPACING} m/unit,  dt={DT} s/step')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ===========================================================================
# 4.  DEBUG / VISUALIZATION  —  Short local trajectory overlay (single frame)
# ===========================================================================

def viz_local_traj(
    action_estfrod_t: np.ndarray,
    frame_idx: int = 0,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (5, 5),
) -> plt.Figure:
    """
    Plot the 8-step local trajectory for a single frame on a metric axes.
    This mirrors what local_traj_image() rasterizes, but as a readable plot.

    Args:
        action_estfrod_t : np.ndarray (8, 4) — single frame.
        frame_idx        : frame index for plot title.
        save_path        : optional output path.

    Returns:
        matplotlib Figure.
    """
    xy_m = action_estfrod_t[:, :2] * TRAIN_METRIC_SPACING   # (8, 2) metres
    xs = np.concatenate([[0], xy_m[:, 0]])
    ys = np.concatenate([[0], xy_m[:, 1]])

    # Heading arrows at each waypoint
    cos_h = action_estfrod_t[:, 2]
    sin_h = action_estfrod_t[:, 3]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ys, xs, '-o', color='royalblue', linewidth=1.5, markersize=4,
            label='local waypoints')
    ax.plot(0, 0, 'go', markersize=10, zorder=5, label='robot origin')

    # Draw heading arrows
    arrow_len = 0.05
    for i in range(N_STEPS):
        ax.annotate('', xy=(ys[i+1] + arrow_len * (-sin_h[i]),
                             xs[i+1] + arrow_len * cos_h[i]),
                    xytext=(ys[i+1], xs[i+1]),
                    arrowprops=dict(arrowstyle='->', color='orange', lw=1.0))

    ax.set_aspect('equal')
    ax.set_xlabel('Y (left, m)')
    ax.set_ylabel('X (forward, m)')
    ax.set_title(f'Local 8-step trajectory  (frame {frame_idx})  '
                 f'[spacing={TRAIN_METRIC_SPACING} m/unit]')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ===========================================================================
# Demo — run against a fake episode to verify shapes and plots
# ===========================================================================

def _sanity_check_yaw_sign():
    """
    Yaw sign sanity check.

    Convention being tested:
        action[..., 3] = sin(yaw)  > 0  →  left turn (CCW, +Y direction)
        action[..., 3] = sin(yaw)  < 0  →  right turn (CW, -Y direction)

    Three deterministic test cases with known expected outcomes:
      A. Pure forward (no turn)     → trajectory is a straight vertical line
      B. Constant left turn (+yaw)  → trajectory curves to the LEFT  (+Y side)
      C. Constant right turn (-yaw) → trajectory curves to the RIGHT (-Y side)

    Prints PASS / FAIL for each case.
    """
    print("\n── Yaw sign sanity check ──────────────────────────────────────────")

    def make_action(fwd_norm, yaw_rad):
        a = np.zeros((N_STEPS, 4), dtype=np.float32)
        a[:, 0] = fwd_norm
        a[:, 1] = 0.0
        a[:, 2] = np.cos(yaw_rad)
        a[:, 3] = np.sin(yaw_rad)
        return a

    def accumulate_8steps(action_t):
        """Roll out 8-step local trajectory from a single frame action."""
        gx, gy, gtheta = 0.0, 0.0, 0.0
        for s in range(N_STEPS):
            fwd_m  = action_t[s, 0] * TRAIN_METRIC_SPACING
            left_m = action_t[s, 1] * TRAIN_METRIC_SPACING
            dtheta = np.arctan2(action_t[s, 3], action_t[s, 2])
            gx     += np.cos(gtheta) * fwd_m - np.sin(gtheta) * left_m
            gy     += np.sin(gtheta) * fwd_m + np.cos(gtheta) * left_m
            gtheta += dtheta
        return gx, gy, gtheta

    # Case A: pure forward
    act_fwd = make_action(fwd_norm=1.0, yaw_rad=0.0)
    ex, ey, _ = accumulate_8steps(act_fwd)
    ok_a = ex > 0.05 and abs(ey) < 0.01
    print(f"  A. Pure forward         endpoint=({ex:.3f}, {ey:.3f}) m  "
          f"→ expect X>0, Y≈0  {'PASS ✓' if ok_a else 'FAIL ✗'}")

    # Case B: constant left turn (+yaw → sin > 0)
    act_left = make_action(fwd_norm=1.0, yaw_rad=+0.15)
    lx, ly, _ = accumulate_8steps(act_left)
    ok_b = lx > 0.0 and ly > 0.01   # ends up forward AND to the left (+Y)
    print(f"  B. Constant left turn   endpoint=({lx:.3f}, {ly:.3f}) m  "
          f"→ expect Y>0 (left)  {'PASS ✓' if ok_b else 'FAIL ✗'}")

    # Case C: constant right turn (-yaw → sin < 0)
    act_right = make_action(fwd_norm=1.0, yaw_rad=-0.15)
    rx, ry, _ = accumulate_8steps(act_right)
    ok_c = rx > 0.0 and ry < -0.01  # ends up forward AND to the right (-Y)
    print(f"  C. Constant right turn  endpoint=({rx:.3f}, {ry:.3f}) m  "
          f"→ expect Y<0 (right) {'PASS ✓' if ok_c else 'FAIL ✗'}")

    overall = all([ok_a, ok_b, ok_c])
    print(f"\n  Overall: {'ALL PASS ✓' if overall else 'SOME FAIL ✗ — check yaw sign convention'}")
    print("────────────────────────────────────────────────────────────────\n")
    return overall


def load_actions_from_pkl(pkl_path: str) -> np.ndarray:
    """
    Convert a gnm_format traj_data.pkl into action_estfrod-compatible array.

    The pkl contains GPS-based ground-truth pos/yaw (NOT MBRA predictions).
    This function converts consecutive GPS poses into the same normalised
    [fwd, left, cos(yaw), sin(yaw)] format that action_estfrod uses, so the
    same visualization functions work on both real GPS data and MBRA outputs.

    Conversion:
        For frame t, we build N_STEPS future relative waypoints using GPS pos/yaw.
        Each waypoint k is the position of frame (t+k+1) expressed in the local
        frame of frame t (rotated by -yaw[t], translated by pos[t]).

        fwd_norm  = local_x / TRAIN_METRIC_SPACING
        left_norm = local_y / TRAIN_METRIC_SPACING
        cos_yaw   = cos(yaw[t+k+1] - yaw[t])
        sin_yaw   = sin(yaw[t+k+1] - yaw[t])

    Args:
        pkl_path : path to traj_data.pkl

    Returns:
        np.ndarray (T, N_STEPS, 4) — same layout as action_estfrod.
        T = number of valid frames (frames where t+N_STEPS < total_frames).
    """
    import pickle

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    pos = np.array(data['pos'], dtype=np.float64)   # (F, 2)  [x, y] in metres
    yaw = np.array(data['yaw'], dtype=np.float64)   # (F,)    radians

    F = len(pos)
    T = F - N_STEPS   # frames for which we can build a full 8-step window
    if T <= 0:
        raise ValueError(f"Episode too short ({F} frames) for N_STEPS={N_STEPS}")

    actions = np.zeros((T, N_STEPS, 4), dtype=np.float32)

    for t in range(T):
        curr_x,   curr_y   = pos[t]
        curr_yaw           = yaw[t]
        cos_c, sin_c       = np.cos(-curr_yaw), np.sin(-curr_yaw)   # rotation into local frame

        for k in range(N_STEPS):
            fut_x, fut_y = pos[t + k + 1]
            fut_yaw      = yaw[t + k + 1]

            # Translate then rotate into current robot frame
            dx = fut_x - curr_x
            dy = fut_y - curr_y
            local_fwd  =  cos_c * dx - sin_c * dy   # robot +X (forward)
            local_left =  sin_c * dx + cos_c * dy   # robot +Y (left)

            rel_yaw = fut_yaw - curr_yaw
            # Normalise to [-pi, pi]
            rel_yaw = (rel_yaw + np.pi) % (2 * np.pi) - np.pi

            actions[t, k, 0] = local_fwd  / TRAIN_METRIC_SPACING
            actions[t, k, 1] = local_left / TRAIN_METRIC_SPACING
            actions[t, k, 2] = np.cos(rel_yaw)
            actions[t, k, 3] = np.sin(rel_yaw)

    return actions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pkl',
        default='/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/'
                'FrodoBots-2K/gnm_format/ride_16446_20240115041752/traj_data.pkl',
        help='Path to gnm_format traj_data.pkl'
    )
    parser.add_argument('--out', default='/tmp/traj_debug', help='Output directory')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    # ── 0. Yaw sign sanity check ─────────────────────────────────────────────
    passed = _sanity_check_yaw_sign()
    assert passed, "Yaw sign sanity check failed."

    # ── Load real GPS trajectory from pkl ────────────────────────────────────
    print(f"Loading: {args.pkl}")
    real_actions = load_actions_from_pkl(args.pkl)
    T = real_actions.shape[0]
    print(f"Loaded {T} frames  →  action shape: {real_actions.shape}")

    # ── 1. Training image (frame 0, real GPS data) ───────────────────────────
    img64  = local_traj_image(real_actions[0], img_px=64,
                              forward_range_m=1.0, lateral_range_m=0.5)
    img128 = local_traj_image(real_actions[0], img_px=128,
                              forward_range_m=1.0, lateral_range_m=0.5)
    cv2.imwrite(str(out_dir / "local_traj_64px.png"),  img64)
    cv2.imwrite(str(out_dir / "local_traj_128px.png"), img128)
    print(f"[1] local_traj images saved → {out_dir}")

    # ── 2. Long future path — all frames ─────────────────────────────────────
    fig = viz_long_future(real_actions, save_path=str(out_dir / "long_future.png"))
    plt.close(fig)
    print(f"[2] long_future.png saved → {out_dir}")

    # ── 3. Current-to-goal — frame 0 → last frame ────────────────────────────
    goal_frame = T - 1
    fig = viz_current_to_goal(real_actions, start_frame=0, goal_frame=goal_frame,
                              save_path=str(out_dir / "current_to_goal.png"))
    plt.close(fig)
    print(f"[3] current_to_goal.png saved  (frame 0 → {goal_frame}) → {out_dir}")

    # ── 4. Single-frame local plot ────────────────────────────────────────────
    fig = viz_local_traj(real_actions[0], frame_idx=0,
                         save_path=str(out_dir / "local_traj_plot.png"))
    plt.close(fig)
    print(f"[4] local_traj_plot.png saved → {out_dir}")

    print(f"\nAll outputs → {out_dir}")
