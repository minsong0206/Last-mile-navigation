"""
make_test_video.py

test set(episode-grouped, 학습에 전혀 안 쓰인 episode)에서 하나의 연속 세그먼트를 골라
current image / trajectory plot / map overlay 3분할 프레임을 프레임마다 렌더링하고
mp4 영상으로 이어붙인다.

실행:
  CUDA_VISIBLE_DEVICES=0 /home/ms/uv-envs/mbra/venv/bin/python scripts/omnivla/make_test_video.py \
      --ckpt checkpoints/omnivla_edge_rides11_odom_12m/best.pth \
      --config config/rides11_finetune_odom_12m.yaml \
      --fps 10 --max_frames 200
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import yaml
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "omnivla"))
sys.path.insert(0, str(REPO_ROOT / "osm_pipeline" / "py"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "omnivla" / "inference"))

from finetune_omnivla_edge import (
    MODEL_PARAMS, prepare_batch, episode_grouped_split, load_model, METRIC_WAYPOINT_SPACING,
)
from rides11_dataset import Rides11Dataset

IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD  = np.array([0.229, 0.224, 0.225])


def denorm(t):
    im = t.detach().cpu().permute(1, 2, 0).numpy() * IMG_STD + IMG_MEAN
    return np.clip(im, 0, 1)


def wp_to_pixel(wp_xy_m, img_px, map_range_m):
    px_per_m = img_px / (2 * map_range_m)
    cx = cy = img_px / 2.0
    px = cx - wp_xy_m[:, 1] * px_per_m
    py = cy - wp_xy_m[:, 0] * px_per_m
    return px, py


def pick_longest_test_segment(dataset, test_idx, rank=0, exclude_episodes=()):
    """test_idx(전역 인덱스 리스트) 중 (ep,seg) 조합별 길이 기준 rank번째로 긴 구간을 찾는다.
    rank=0 → 가장 긴 것, rank=1 → 두 번째로 긴 것, ...
    exclude_episodes에 포함된 episode는 후보에서 제외."""
    from collections import defaultdict
    by_seg = defaultdict(list)
    for gi in test_idx:
        ep, seg, fi, seg_local_idx = dataset.valid_samples[gi]
        if ep in exclude_episodes:
            continue
        by_seg[(ep, seg)].append((fi, gi))
    ranked = sorted(by_seg.items(), key=lambda kv: -len(kv[1]))
    best_key, best_list = ranked[rank]
    best_list.sort(key=lambda t: t[0])  # fi 순서로 정렬
    return best_key, [gi for _, gi in best_list]


def render_frame_png(obs_img, map_img, pred_m, gt_m, map_range_m, title=""):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(denorm(obs_img))
    axes[0].axis("off")
    axes[0].set_title(f"Current Image\n{title}", fontsize=9)

    ax = axes[1]
    gt_x, gt_y = -gt_m[:, 1], gt_m[:, 0]
    pr_x, pr_y = -pred_m[:, 1], pred_m[:, 0]
    ax.plot(np.insert(gt_x, 0, 0), np.insert(gt_y, 0, 0), '-o', color='green', lw=2, ms=6, label='GT')
    ax.plot(np.insert(pr_x, 0, 0), np.insert(pr_y, 0, 0), '-o', color='red', lw=2, ms=6, label='Pred')
    ax.plot(0, 0, 'b*', ms=12)
    ax.set_xlim(-3, 3); ax.set_ylim(-0.1, 10)
    ax.set_xlabel("Left/Right (m)"); ax.set_ylabel("Forward (m)")
    ax.set_title("Trajectory", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_aspect('equal')

    ax = axes[2]
    map_np = denorm(map_img)
    ax.imshow(map_np)
    img_px = map_np.shape[0]
    gpx, gpy = wp_to_pixel(gt_m, img_px, map_range_m)
    ppx, ppy = wp_to_pixel(pred_m, img_px, map_range_m)
    ax.plot(gpx, gpy, '--', color='lime', lw=2, label='GT')
    ax.plot(ppx, ppy, '-', color='red', lw=2.5, label='Pred')
    ax.plot(img_px / 2, img_px / 2, 'w*', ms=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Map Overlay (range={map_range_m:.0f}m)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--max_frames", type=int, default=200)
    p.add_argument("--out", type=str, default=str(REPO_ROOT / "attention_analysis" / "test_set_video.mp4"))
    p.add_argument("--rank", type=int, default=0, help="0=가장 긴 test 세그먼트, 1=두 번째로 긴 것 ...")
    p.add_argument("--exclude_episodes", type=int, nargs="*", default=[])
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    map_range_m = cfg.get("map_range_m", 25.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.ckpt, MODEL_PARAMS, device)
    model.to(device).eval()

    dataset = Rides11Dataset(
        arrow_path=cfg["arrow_path"], scores_path=cfg["scores_path"],
        osm_root=cfg["osm_root"], video_root=cfg["video_root"],
    )
    _, _, test_ds = episode_grouped_split(dataset, cfg.get("val_ratio", 0.1), cfg.get("test_ratio", 0.1), seed=42)

    seg_key, ordered_idx = pick_longest_test_segment(
        dataset, test_ds.indices, rank=args.rank, exclude_episodes=set(args.exclude_episodes))
    ep, seg = seg_key
    print(f"[video] longest test segment: episode={ep} seg={seg}  n_frames={len(ordered_idx)}")
    ordered_idx = ordered_idx[:args.max_frames]
    print(f"[video] rendering {len(ordered_idx)} frames (max_frames={args.max_frames})")

    frames = []
    for i, gi in enumerate(ordered_idx):
        sample = dataset[gi]
        batch = [sample]
        inp = prepare_batch(
            {k: torch.stack([b[k] for b in batch]) if torch.is_tensor(batch[0][k])
             else [b[k] for b in batch] for k in batch[0]},
            device,
        )
        with torch.no_grad():
            pred, _, _ = model(
                inp["obs_img"], inp["goal_pose"], inp["map_images"],
                inp["goal_img"], inp["goal_mask"], inp["feat_text"], inp["current_img"],
            )
        pred_m = pred[0, :, :2].cpu().numpy() * METRIC_WAYPOINT_SPACING
        gt_m = inp["gt_waypoints"][0].cpu().numpy() * METRIC_WAYPOINT_SPACING

        fi = sample["frame_index"].item() if torch.is_tensor(sample["frame_index"]) else sample["frame_index"]
        title = f"ep{ep:04d}_seg{seg:02d}_fi{fi:06d}  [{i+1}/{len(ordered_idx)}]"
        buf = render_frame_png(inp["obs_img"][0, -3:], inp["map_images"][0], pred_m, gt_m, map_range_m, title)
        frames.append(buf)
        if (i + 1) % 20 == 0:
            print(f"  rendered {i+1}/{len(ordered_idx)}")

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"[video] saved: {args.out}  ({len(frames)} frames @ {args.fps}fps = {len(frames)/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
