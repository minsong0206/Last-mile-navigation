"""
test_map_causality.py

세 가지를 확인:
  A) held-out test 확인 — finetune_omnivla_edge.py와 동일한 seed=42 random_split으로
     실제로 학습에 안 쓰인 test_ds에서만 map ablation(zeros/noise/shuffled) 수행
  B) ablation 결과 시각화 — normal vs zeros/noise 조건별 예측 궤적을
     map 없이 plain ego-frame 그래프(동일 축척)로 겹쳐 그려 차이를 쉽게 비교
  C) straight vs curve map swap test — output_rides_00 (fine-tuning에 안 쓰인 별도 라이드)의
     OSM map을 gps.csv 기반 곡률로 straight/curve 분류 후, held-out 샘플의 obs는
     고정하고 map만 교체하며 예측 궤적이 실제로 달라지는지 확인
  D) matched obs+map test — output_rides_00의 실제 카메라 프레임과 그 위치의 map을
     함께 매칭해서 넣었을 때 (새 환경/새 카메라에서의 realistic 동작 확인).
     rides_00(episode 0~251, ride_16xxx~17xxx, 2024-01~02)은 rides_11 fine-tuning에
     쓰인 raw 데이터(ride_28xxx, 2024-04)와 ride ID/날짜가 겹치지 않는 완전히 별도 촬영분.
  E) matched obs+map SEQUENCE — straight 세그먼트 1개, curve 세그먼트 1개에 대해
     한 episode 안에서 시간에 따라 camera/predicted trajectory/map을 연속으로 나열해서
     모델이 실제 도로 형태를 따라가는지, 아니면 map을 약한 신호로만 쓰는지 확인.

실행 (host, mbra venv):
  /home/ms/uv-envs/mbra/venv/bin/python scripts/analysis/test_map_causality.py --method all
  /home/ms/uv-envs/mbra/venv/bin/python scripts/analysis/test_map_causality.py --method heldout
  /home/ms/uv-envs/mbra/venv/bin/python scripts/analysis/test_map_causality.py --method swap
  /home/ms/uv-envs/mbra/venv/bin/python scripts/analysis/test_map_causality.py --method matched
  /home/ms/uv-envs/mbra/venv/bin/python scripts/analysis/test_map_causality.py --method sequence
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "omnivla" / "inference"))
from model_omnivla_edge_odom import OmniVLA_edge_odom

sys.path.insert(0, str(REPO_ROOT / "osm_pipeline" / "py"))
from rides11_dataset import Rides11Dataset, METRIC_WAYPOINT_SPACING, IMG_MEAN, IMG_STD, VideoReader

sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
import analyze_dataset_distribution as dd  # gps -> curvature/scenario 분류 재사용

BASE       = Path("/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA")
CKPT       = BASE / "checkpoints/omnivla_edge_rides11_odom/best.pth"
ARROW      = BASE / "FrodoBots-2K/processed/output_rides_11/train/data-00000-of-00001.arrow"
SCORES     = BASE / "osm_pipeline/osm_data/output_rides_11/episode_scores.json"
OSM_ROOT   = BASE / "osm_pipeline/osm_data/output_rides_11/osm_maps_arrow"
VID_ROOT   = BASE / "FrodoBots-2K/processed/output_rides_11"
RIDES00_OSM_ROOT = BASE / "osm_pipeline/osm_data/output_rides_00/osm_maps"
RAW_DATASET_ROOT = BASE / "FrodoBots-2K/processed/frodobots_dataset"  # rides_00 GPS/영상의 원본 raw 데이터셋
RAW_ZARR_PATH    = RAW_DATASET_ROOT / "dataset_cache.zarr"
OUT_DIR    = BASE / "attention_analysis"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAP_RANGE_M = 25.0  # osm_map_generator.py 기준, half-width in meters

MODEL_PARAMS = dict(
    context_size=5, len_traj_pred=8, learn_angle=True,
    obs_encoder="efficientnet-b0", obs_encoding_size=1024,
    late_fusion=False, mha_num_attention_heads=4,
    mha_num_attention_layers=4, mha_ff_dim_factor=4,
)


# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path):
    model = OmniVLA_edge_odom(**MODEL_PARAMS)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model.to(DEVICE).eval()
    print(f"Loaded checkpoint: {ckpt_path}")
    return model


def get_test_split(seed=42, val_ratio=0.1, test_ratio=0.1):
    """finetune_omnivla_edge.py의 main()과 완전히 동일한 split 로직 재현.
    이 test_ds에 속한 샘플만 fine-tuning 중 절대 gradient를 본 적이 없음."""
    dataset = Rides11Dataset(str(ARROW), str(SCORES), str(OSM_ROOT), str(VID_ROOT))
    val_size = int(len(dataset) * val_ratio)
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - val_size - test_size
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"[split] dataset={len(dataset):,}  train={len(train_ds):,}  "
          f"val={len(val_ds):,}  test={len(test_ds):,}")
    return dataset, train_ds, val_ds, test_ds


def make_inputs(batch, device):
    obs_stack  = torch.stack([b["obs_stack"]    for b in batch]).to(device)
    map_images = torch.stack([b["map_images"]   for b in batch]).to(device)
    gt_wp      = torch.stack([b["gt_waypoints"] for b in batch]).to(device)
    B = obs_stack.shape[0]
    obs_cur   = obs_stack[:, -3:]
    goal_pose = torch.zeros(B, 4, device=device)
    goal_img  = obs_cur
    goal_mask = torch.zeros(B, dtype=torch.long, device=device)
    feat_text = torch.zeros(B, 512, device=device)
    cur_img   = F.interpolate(obs_cur, (224, 224), mode='bilinear', align_corners=False)
    return obs_stack, goal_pose, map_images, goal_img, goal_mask, feat_text, cur_img, gt_wp


def ade_fde(pred, gt):
    p = pred.detach().cpu().numpy() * METRIC_WAYPOINT_SPACING
    g = gt.detach().cpu().numpy()   * METRIC_WAYPOINT_SPACING
    d = np.sqrt(((p - g) ** 2).sum(-1))
    return d.mean(-1).mean(), d[:, -1].mean()


def denorm_img(t):
    mean = np.array(IMG_MEAN); std = np.array(IMG_STD)
    im = t.detach().cpu().permute(1, 2, 0).numpy() * std + mean
    return np.clip(im, 0, 1)


def wp_to_pixel(wp_xy_m, img_px=96, map_range_m=MAP_RANGE_M):
    """ego-frame waypoint(x=forward, y=left), meters → OSM map pixel 좌표.
    osm_map_generator.py / finetune_omnivla_edge.py의 overlay 규약과 동일:
    ego(중심)에서 위쪽이 forward(+x), 왼쪽이 +y."""
    px_per_m = img_px / (2 * map_range_m)
    cx = cy = img_px / 2.0
    px = cx - wp_xy_m[:, 1] * px_per_m
    py = cy - wp_xy_m[:, 0] * px_per_m
    return px, py


def to_ego_plot_xy(wp_xy_m):
    """ego-frame waypoint(x=forward, y=left), meters → plain plot (x=lateral, y=forward).
    오른쪽으로 꺾으면 +x, 왼쪽으로 꺾으면 -x가 되도록: plot_x = -y(left), plot_y = x(forward)."""
    return -wp_xy_m[:, 1], wp_xy_m[:, 0]


# ─────────────────────────────────────────────────────────────────────────────
# A + B) Held-out ablation test + visualization
# ─────────────────────────────────────────────────────────────────────────────

def run_heldout_ablation(model, n_batches=20, bs=16, n_viz=6):
    print("\n" + "=" * 60)
    print("A) HELD-OUT TEST SET ABLATION (seed=42 split, test_ds only)")
    print("=" * 60)

    _, _, _, test_ds = get_test_split()
    rng = np.random.default_rng(0)
    order = rng.permutation(len(test_ds))[: n_batches * bs]

    results = {k: {"ade": [], "fde": []} for k in ["normal", "zeros", "noise", "shuffled"]}
    viz_batch = None

    for i in range(n_batches):
        idx = order[i * bs:(i + 1) * bs]
        batch = [test_ds[int(j)] for j in idx]
        obs, gp, map_img, gi, gm, ft, ci, gt = make_inputs(batch, DEVICE)

        with torch.no_grad():
            preds = {}
            pred, _, _ = model(obs, gp, map_img, gi, gm, ft, ci)
            preds["normal"] = pred
            a, f = ade_fde(pred[:, :, :2], gt); results["normal"]["ade"].append(a); results["normal"]["fde"].append(f)

            pred_z, _, _ = model(obs, gp, torch.zeros_like(map_img), gi, gm, ft, ci)
            preds["zeros"] = pred_z
            a, f = ade_fde(pred_z[:, :, :2], gt); results["zeros"]["ade"].append(a); results["zeros"]["fde"].append(f)

            pred_n, _, _ = model(obs, gp, torch.randn_like(map_img), gi, gm, ft, ci)
            preds["noise"] = pred_n
            a, f = ade_fde(pred_n[:, :, :2], gt); results["noise"]["ade"].append(a); results["noise"]["fde"].append(f)

            perm = torch.randperm(map_img.shape[0])
            pred_s, _, _ = model(obs, gp, map_img[perm], gi, gm, ft, ci)
            preds["shuffled"] = pred_s
            a, f = ade_fde(pred_s[:, :, :2], gt); results["shuffled"]["ade"].append(a); results["shuffled"]["fde"].append(f)

        if i == 0:
            viz_batch = dict(obs=obs, map_img=map_img, gt=gt, preds=preds)

    print(f"\n{'Condition':<12}  {'ADE (m)':>10}  {'FDE (m)':>10}  {'ΔADE vs normal':>16}")
    ade_normal = np.mean(results["normal"]["ade"]); fde_normal = np.mean(results["normal"]["fde"])
    print(f"{'normal':<12}  {ade_normal:10.4f}  {fde_normal:10.4f}  {'baseline':>16}")
    for cond in ["zeros", "noise", "shuffled"]:
        a = np.mean(results[cond]["ade"]); f = np.mean(results[cond]["fde"])
        delta = a - ade_normal; pct = delta / ade_normal * 100
        print(f"{'map=' + cond:<12}  {a:10.4f}  {f:10.4f}  {delta:>+12.4f}m ({pct:+.1f}%)")

    delta_zeros = np.mean(results["zeros"]["ade"]) - ade_normal
    if delta_zeros > 0.01:
        print(f"\n  ✓ HELD-OUT 데이터에서도 MAP이 실제로 쓰이고 있음: map=zeros가 ADE를 {delta_zeros:+.4f}m 악화시킴")
    else:
        print(f"\n  ⚠ HELD-OUT 데이터에서 MAP 영향 미미: ΔADE {delta_zeros:+.4f}m")

    # ── bar chart ──
    conds = ["normal", "zeros", "noise", "shuffled"]
    ades = [np.mean(results[c]["ade"]) for c in conds]
    colors = ["#4CAF50", "#9E9E9E", "#F44336", "#FF9800"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(conds, ades, color=colors, edgecolor='white')
    ax.axhline(ades[0], color='green', ls='--', lw=1, alpha=0.5)
    ax.set_ylabel("ADE (m)")
    ax.set_title(f"Held-out test set ablation (n={n_batches*bs})\n(map을 교체했을 때 ADE 변화, seed=42 split)")
    for bar, v in zip(bars, ades):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002, f"{v:.4f}", ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "heldout_ablation_bar.png", dpi=150)
    plt.close(fig)
    print(f"  → Saved: {OUT_DIR}/heldout_ablation_bar.png")

    # ── B) per-sample visualization: normal vs zeros/noise predicted trajectory,
    #      plain ego-frame plot (map-scale 때문에 차이가 안 보이던 문제 해결) ──
    plain_conds = ["normal", "zeros", "noise"]
    colors = {"normal": "#2E7D32", "zeros": "#9E9E9E", "noise": "#E53935"}
    n_viz = min(n_viz, viz_batch["obs"].shape[0])

    # 카메라 이미지(참고용) + 궤적 플롯, 샘플당 2열
    fig, axes = plt.subplots(n_viz, 2, figsize=(8, 3.2 * n_viz))
    if n_viz == 1:
        axes = axes[np.newaxis, :]

    # 전체 샘플에 걸쳐 공통 축 범위 계산 (줌인 + 일관된 스케일)
    all_x, all_y = [], []
    for row in range(n_viz):
        gt_m = viz_batch["gt"][row].cpu().numpy() * METRIC_WAYPOINT_SPACING
        gx, gy = to_ego_plot_xy(gt_m)
        all_x.append(gx); all_y.append(gy)
        for cond in plain_conds:
            p = viz_batch["preds"][cond][row, :, :2].detach().cpu().numpy() * METRIC_WAYPOINT_SPACING
            px, py = to_ego_plot_xy(p)
            all_x.append(px); all_y.append(py)
    all_x = np.concatenate(all_x); all_y = np.concatenate(all_y)
    xmax = max(0.4, np.abs(all_x).max() * 1.25)
    ymin = min(0.0, all_y.min()) - 0.15
    ymax = all_y.max() * 1.15 + 0.15

    for row in range(n_viz):
        cur_img = denorm_img(viz_batch["obs"][row, -3:])
        axes[row, 0].imshow(cur_img)
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
        axes[row, 0].set_ylabel(f"sample {row}", fontsize=9)
        if row == 0:
            axes[row, 0].set_title("current obs", fontsize=9)

        ax = axes[row, 1]
        gt_m = viz_batch["gt"][row].cpu().numpy() * METRIC_WAYPOINT_SPACING
        gx, gy = to_ego_plot_xy(gt_m)
        ax.plot(gx, gy, 'k--o', ms=4, lw=1.5, label="GT", zorder=5)
        for cond in plain_conds:
            p = viz_batch["preds"][cond][row, :, :2].detach().cpu().numpy() * METRIC_WAYPOINT_SPACING
            px, py = to_ego_plot_xy(p)
            ax.plot(px, py, '-o', ms=3.5, lw=2, color=colors[cond], label=f"map={cond}")
        ax.plot(0, 0, 'b*', ms=10, zorder=6)
        ax.axhline(0, color='lightgray', lw=0.6, zorder=0)
        ax.axvline(0, color='lightgray', lw=0.6, zorder=0)
        ax.set_xlim(-xmax, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.set_xlabel("lateral (m)", fontsize=8)
        ax.set_ylabel("forward (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        if row == 0:
            ax.set_title("predicted trajectory (ego frame)", fontsize=9)
            ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    fig.suptitle("Held-out ablation: predicted trajectory under different map conditions\n"
                 "(검정 점선=GT, 파란 별=ego, 동일 축척)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.85, 0.95])
    fig.savefig(OUT_DIR / "heldout_ablation_traj_plain.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Saved: {OUT_DIR}/heldout_ablation_traj_plain.png")


# ─────────────────────────────────────────────────────────────────────────────
# C) straight vs curve map swap test using output_rides_00
# ─────────────────────────────────────────────────────────────────────────────

def classify_rides00_segments():
    """output_rides_00/osm_maps/episode_*_seg*/gps.csv 를 curvature로 분류.

    주의: 원래는 raw lat/lon(GPS)으로 heading을 계산했으나, 저속/정지 구간에서
    GPS 노이즈만으로도 수백 deg의 가짜 heading change가 나오는 것을 확인함
    (예: episode_0008_seg06 — raw GPS 기준 net=-301°, 그러나 EKF로 스무딩된
    filtered_heading 기준 net=+5.9°, 즉 실제로는 거의 직진).
    그래서 raw lat/lon 대신 frodobots_dataset zarr의 EKF-smoothed
    filtered_position(로컬 xy, m)을 사용해 curvature를 계산한다.
    """
    seg_dirs = sorted(RIDES00_OSM_ROOT.glob("episode_*_seg*"))
    print(f"[rides_00] {len(seg_dirs)} segments found under {RIDES00_OSM_ROOT}")

    import csv
    raw_cache = {}
    stats_list = []
    for seg_dir in seg_dirs:
        name = seg_dir.name  # episode_0152_seg04
        try:
            ep = int(name.split("_")[1])
            seg = int(name.split("seg")[1])
        except (IndexError, ValueError):
            continue

        lats, lons = dd.read_gps(seg_dir)
        if lats is None or len(lats) < 20:
            continue

        if ep not in raw_cache:
            raw_cache[ep] = _load_raw_episode_table(ep)
        raw = raw_cache[ep]
        if raw is None:
            continue

        offset = _find_segment_offset(lats, lons, raw["lat"], raw["lon"])
        n_local = len(lats)
        if offset + n_local > len(raw["filtered_pos"]):
            n_local = len(raw["filtered_pos"]) - offset
        if n_local < 20:
            continue
        fp_seg = raw["filtered_pos"][offset:offset + n_local]  # (n,2) EKF-smoothed local xy, m
        x, y = fp_seg[:, 0], fp_seg[:, 1]

        dists = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        traj_len = float(dists.sum())
        if traj_len < 5.0:
            continue
        rx, ry = dd.resample_by_distance(x, y, dd.DIST_STEP_M)
        if len(rx) < 4:
            continue
        headings = dd.compute_headings(rx, ry)
        changes = dd.compute_heading_changes(headings, dd.SMOOTH_WINDOW)
        events = dd.detect_turn_events(changes, dd.TURN_EVENT_THRESH)
        total_heading = float(np.abs(changes).sum())
        net_heading = float(changes.sum())
        max_L = max((abs(e["total_deg"]) for e in events if e["direction"] == "L"), default=0.0)
        max_R = max((abs(e["total_deg"]) for e in events if e["direction"] == "R"), default=0.0)
        n_L = sum(1 for e in events if e["direction"] == "L")
        n_R = sum(1 for e in events if e["direction"] == "R")
        stats = {
            "seg_key": name, "seg_dir": seg_dir, "n_frames": n_local,
            "traj_len_m": traj_len, "total_heading_deg": total_heading,
            "net_heading_deg": net_heading, "max_left_turn_deg": max_L,
            "max_right_turn_deg": max_R, "n_turn_events": len(events),
            "n_left_events": n_L, "n_right_events": n_R,
            "curvature_density": total_heading / traj_len * 100,
        }
        stats["scenario"] = dd.classify_segment(stats)
        stats_list.append(stats)

    from collections import Counter
    print("[rides_00] scenario counts (filtered_heading 기반):",
          Counter(s["scenario"] for s in stats_list))
    return stats_list


def load_map_tensor(png_path, device):
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])
    img = Image.open(png_path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


def pick_map_png(seg_stat, frac=0.6):
    """세그먼트 내 대표 프레임 하나를 골라 osm_map_XXXXXX.png 경로 반환.
    frac=0.6 → 회전/커브가 진행 중인 지점(세그먼트 중후반)을 선호."""
    pngs = sorted(seg_stat["seg_dir"].glob("osm_map_*.png"))
    if not pngs:
        return None
    idx = int(len(pngs) * frac)
    idx = min(idx, len(pngs) - 1)
    return pngs[idx]


def run_map_swap_test(model, n_each=3):
    print("\n" + "=" * 60)
    print("C) STRAIGHT vs CURVE MAP SWAP TEST (output_rides_00, obs 고정)")
    print("=" * 60)

    stats_list = classify_rides00_segments()
    straight = [s for s in stats_list if s["scenario"] == "straight"]
    curve = [s for s in stats_list
             if s["scenario"] in ("curve_left", "curve_right", "sharp_left_turn", "sharp_right_turn")]

    straight = sorted(straight, key=lambda s: s["curvature_density"])[:n_each]
    curve = sorted(curve, key=lambda s: -s["curvature_density"])[:n_each]

    print(f"[select] straight x{len(straight)}, curve x{len(curve)}")
    for s in straight:
        print(f"    straight: {s['seg_key']}  curv_density={s['curvature_density']:.1f} deg/100m")
    for s in curve:
        print(f"    curve   : {s['seg_key']}  scenario={s['scenario']}  "
              f"net={s['net_heading_deg']:.1f}deg  curv_density={s['curvature_density']:.1f} deg/100m")

    selected = [("straight", s) for s in straight] + [("curve", s) for s in curve]
    if not selected:
        print("  ⚠ 조건에 맞는 rides_00 세그먼트를 찾지 못함")
        return

    # 고정 obs: held-out test_ds의 한 샘플 (실제 fine-tuning에서 본 적 없는 obs)
    _, _, _, test_ds = get_test_split()
    fixed_sample = test_ds[0]
    fixed_gt_m = fixed_sample["gt_waypoints"].numpy() * METRIC_WAYPOINT_SPACING  # (8,2)
    obs = fixed_sample["obs_stack"].unsqueeze(0).to(DEVICE)
    obs_cur = obs[:, -3:]
    goal_pose = torch.zeros(1, 4, device=DEVICE)
    goal_img = obs_cur
    goal_mask = torch.zeros(1, dtype=torch.long, device=DEVICE)
    feat_text = torch.zeros(1, 512, device=DEVICE)
    cur_img = F.interpolate(obs_cur, (224, 224), mode='bilinear', align_corners=False)

    cmap_by_label = {"straight": plt.cm.Blues, "curve": plt.cm.Reds}
    seen_count = {"straight": 0, "curve": 0}

    all_pred_xy = []
    cases = []  # {label, seg_key, pred_m, map_np, color}
    for label, seg_stat in selected:
        png_path = pick_map_png(seg_stat)
        if png_path is None:
            continue
        map_tensor = load_map_tensor(png_path, DEVICE)

        with torch.no_grad():
            pred, _, _ = model(obs, goal_pose, map_tensor, goal_img, goal_mask, feat_text, cur_img)
        pred_m = pred[0, :, :2].detach().cpu().numpy() * METRIC_WAYPOINT_SPACING  # (8,2)
        all_pred_xy.append((label, seg_stat["seg_key"], pred_m))

        n = seen_count[label]; seen_count[label] += 1
        color = cmap_by_label[label](0.45 + 0.18 * n)
        cases.append({
            "label": label, "seg_key": seg_stat["seg_key"], "pred_m": pred_m,
            "map_np": denorm_img(map_tensor[0]), "color": color,
        })

    if not cases:
        print("  ⚠ 조건에 맞는 rides_00 세그먼트를 찾지 못함")
        return

    # ── 하나의 그림: 왼쪽 = GT+예측 궤적 overlay, 오른쪽 = 각 궤적에 대응하는 map ──
    n_maps = len(cases)
    n_cols_map = 3
    n_rows_map = int(np.ceil(n_maps / n_cols_map))
    fig = plt.figure(figsize=(7 + 2.3 * n_cols_map, max(6.5, 2.3 * n_rows_map)))
    gs = fig.add_gridspec(n_rows_map, n_cols_map + 3, wspace=0.35, hspace=0.45)

    ax_traj = fig.add_subplot(gs[:, :3])
    gx, gy = to_ego_plot_xy(fixed_gt_m)
    ax_traj.plot(gx, gy, 'k--o', ms=5, lw=2, label="GT (fixed obs)", zorder=6)
    all_x, all_y = list(gx), list(gy)
    for c in cases:
        px, py = to_ego_plot_xy(c["pred_m"])
        ax_traj.plot(px, py, '-o', ms=4, lw=2, color=c["color"],
                     label=f"{c['label']}: {c['seg_key']}")
        all_x += list(px); all_y += list(py)
    all_x = np.array(all_x); all_y = np.array(all_y)
    xmax = max(0.4, np.abs(all_x).max() * 1.25)
    ymin = min(0.0, all_y.min()) - 0.15
    ymax = all_y.max() * 1.15 + 0.15
    ax_traj.plot(0, 0, 'b*', ms=12, zorder=7)
    ax_traj.axhline(0, color='lightgray', lw=0.6, zorder=0)
    ax_traj.axvline(0, color='lightgray', lw=0.6, zorder=0)
    ax_traj.set_xlim(-xmax, xmax); ax_traj.set_ylim(ymin, ymax)
    ax_traj.set_aspect('equal')
    ax_traj.set_xlabel("lateral (m)"); ax_traj.set_ylabel("forward (m)")
    ax_traj.set_title("predicted trajectory (ego frame, obs fixed)", fontsize=11)
    ax_traj.legend(fontsize=7, loc="upper left", bbox_to_anchor=(-0.02, -0.08), ncol=2)

    for i, c in enumerate(cases):
        r, col = divmod(i, n_cols_map)
        ax_m = fig.add_subplot(gs[r, 3 + col])
        ax_m.imshow(c["map_np"])
        px, py = wp_to_pixel(c["pred_m"])
        ax_m.plot(px, py, '-', lw=2.5, color=c["color"])
        ax_m.plot(48, 48, 'w*', ms=9)
        ax_m.set_xticks([]); ax_m.set_yticks([])
        for spine in ax_m.spines.values():
            spine.set_edgecolor(c["color"]); spine.set_linewidth(3)
        ax_m.set_title(f"{c['label']}\n{c['seg_key']}", fontsize=8)

    fig.suptitle("Map swap test: obs 고정, map만 straight/curve로 교체했을 때\n"
                 "예측 궤적이 실제로 달라지는가? (output_rides_00, fine-tuning에 미사용 데이터)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_DIR / "rides00_map_swap_combined.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Saved: {OUT_DIR}/rides00_map_swap_combined.png")

    # 수치 비교: straight 그룹 평균 궤적 vs curve 그룹 평균 궤적의 net lateral offset
    straight_lat = [p[:, 1].sum() for lbl, _, p in all_pred_xy if lbl == "straight"]
    curve_lat = [p[:, 1].sum() for lbl, _, p in all_pred_xy if lbl == "curve"]
    if straight_lat and curve_lat:
        print(f"\n  straight map → 평균 누적 lateral offset: {np.mean(straight_lat):+.3f} m")
        print(f"  curve map    → 평균 누적 lateral offset: {np.mean(curve_lat):+.3f} m")
        print(f"  (obs가 동일하므로 이 차이는 순수하게 map 입력 때문)")


# ─────────────────────────────────────────────────────────────────────────────
# D) matched obs+map test — rides_00의 실제 카메라 프레임 + 그 위치의 map을 함께 사용
# ─────────────────────────────────────────────────────────────────────────────

def _load_raw_episode_table(episode: int):
    """frodobots_dataset zarr에서 해당 episode의 (frame_index, timestamp, lat, lon, video_path)를
    frame_index 기준 정렬하여 반환."""
    import zarr
    z = zarr.open(str(RAW_ZARR_PATH), mode="r")
    ep_arr = z["episode_index"][:]
    mask = ep_arr == episode
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return None
    fi = z["frame_index"][:][idxs]
    order = np.argsort(fi)
    idxs = idxs[order]
    fi = fi[order]
    ts = z["observation.images.front.timestamp"][:][idxs]
    lat = z["observation.latitude"][:][idxs]
    lon = z["observation.longitude"][:][idxs]
    filtered_pos = z["observation.filtered_position"][:][idxs]  # EKF-smoothed local xy (m)
    filtered_heading = z["observation.filtered_heading"][:][idxs]  # radians
    video_rel = str(z["observation.images.front.path"][idxs[0]])
    return {"frame_index": fi, "timestamp": ts, "lat": lat, "lon": lon,
            "filtered_pos": filtered_pos, "filtered_heading": filtered_heading,
            "video_path": str(RAW_DATASET_ROOT / video_rel)}


def _find_segment_offset(seg_lat, seg_lon, ep_lat, ep_lon):
    """segment(로컬)의 GPS 시퀀스가 전체 episode GPS 시퀀스 어디서 시작하는지
    슬라이딩 윈도우 L2 오차 최소화로 탐색."""
    n_seg, n_ep = len(seg_lat), len(ep_lat)
    if n_ep < n_seg:
        return 0
    best_off, best_err = 0, float("inf")
    for off in range(0, n_ep - n_seg + 1):
        d_lat = ep_lat[off:off + n_seg] - seg_lat
        d_lon = ep_lon[off:off + n_seg] - seg_lon
        err = float(np.sum(d_lat ** 2 + d_lon ** 2))
        if err < best_err:
            best_err, best_off = err, off
    return best_off


def run_matched_obs_map_test(model, n_each=2):
    print("\n" + "=" * 60)
    print("D) MATCHED OBS+MAP TEST (output_rides_00, 실제 카메라+map 동시 사용)")
    print("=" * 60)

    stats_list = classify_rides00_segments()
    straight = sorted([s for s in stats_list if s["scenario"] == "straight"],
                       key=lambda s: s["curvature_density"])[:n_each]
    curve = sorted([s for s in stats_list if s["scenario"] in
                    ("curve_left", "curve_right", "sharp_left_turn", "sharp_right_turn")],
                   key=lambda s: -s["curvature_density"])[:n_each]
    selected = [("straight", s) for s in straight] + [("curve", s) for s in curve]
    print(f"[select] straight x{len(straight)}, curve x{len(curve)}")

    obs_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])
    video_reader = VideoReader()

    CTX_STRIDE, N_CTX = 3, 5
    PAST_MARGIN = CTX_STRIDE * N_CTX  # 15

    results = []
    for label, seg_stat in selected:
        ep = int(seg_stat["seg_key"].split("_")[1])
        raw = _load_raw_episode_table(ep)
        if raw is None:
            print(f"  ⚠ {seg_stat['seg_key']}: raw episode {ep} not found, skip")
            continue

        # segment gps.csv (로컬 프레임) → 전체 episode gps 시퀀스에서의 offset 탐색
        import csv
        with open(seg_stat["seg_dir"] / "gps.csv") as f:
            rows = list(csv.DictReader(f))
        seg_lat = np.array([float(r["latitude"]) for r in rows])
        seg_lon = np.array([float(r["longitude"]) for r in rows])
        offset = _find_segment_offset(seg_lat, seg_lon, raw["lat"], raw["lon"])

        # 대표 프레임: 세그먼트 60% 지점 (커브 진행 중인 지점을 선호)
        local_idx = min(int(len(rows) * 0.6), len(rows) - 1)
        global_row = offset + local_idx
        if global_row < PAST_MARGIN or global_row >= len(raw["frame_index"]):
            global_row = max(PAST_MARGIN, min(global_row, len(raw["frame_index"]) - 1))

        # context 5장 + 현재 1장 timestamp
        ctx_rows = [global_row - CTX_STRIDE * (N_CTX - k) for k in range(N_CTX)] + [global_row]
        ctx_rows = [max(0, r) for r in ctx_rows]
        imgs = []
        for r in ctx_rows:
            ts = float(raw["timestamp"][r])
            pil = video_reader.get_frame(raw["video_path"], ts)
            imgs.append(obs_transform(pil))
        obs_stack = torch.cat(imgs, dim=0).unsqueeze(0).to(DEVICE)  # (1,18,96,96)
        obs_cur = obs_stack[:, -3:]

        map_png = pick_map_png(seg_stat, frac=0.6)
        map_tensor = load_map_tensor(map_png, DEVICE)

        goal_pose = torch.zeros(1, 4, device=DEVICE)
        goal_img = obs_cur
        goal_mask = torch.zeros(1, dtype=torch.long, device=DEVICE)
        feat_text = torch.zeros(1, 512, device=DEVICE)
        cur_img = F.interpolate(obs_cur, (224, 224), mode='bilinear', align_corners=False)

        with torch.no_grad():
            pred, _, _ = model(obs_stack, goal_pose, map_tensor, goal_img, goal_mask, feat_text, cur_img)
        pred_m = pred[0, :, :2].detach().cpu().numpy() * METRIC_WAYPOINT_SPACING

        results.append({
            "label": label, "seg_key": seg_stat["seg_key"], "pred_m": pred_m,
            "cam_img": denorm_img(obs_stack[0, -3:]), "map_img": denorm_img(map_tensor[0]),
        })
        print(f"    {label:8s} {seg_stat['seg_key']}: offset={offset} global_row={global_row}  "
              f"pred lateral(sum)={pred_m[:,1].sum():+.3f}m")

    video_reader.close_all()

    if not results:
        print("  ⚠ 매칭된 샘플 없음")
        return

    fig, axes = plt.subplots(2, len(results), figsize=(3 * len(results), 6))
    if len(results) == 1:
        axes = axes[:, np.newaxis]
    for col, r in enumerate(results):
        axes[0, col].imshow(r["cam_img"])
        axes[0, col].set_title(f"{r['label']}\n{r['seg_key']} (real cam)", fontsize=8)
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])

        px, py = wp_to_pixel(r["pred_m"])
        axes[1, col].imshow(r["map_img"])
        axes[1, col].plot(px, py, 'r-', lw=2)
        axes[1, col].plot(48, 48, 'w*', markersize=10)
        axes[1, col].set_xticks([]); axes[1, col].set_yticks([])
        axes[1, col].set_title("map + predicted traj", fontsize=8)

    fig.suptitle("Matched obs+map test (output_rides_00, fine-tuning에 미사용 raw 데이터)\n"
                 "실제 카메라+map을 함께 넣었을 때의 예측 (real-world generalization check)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT_DIR / "rides00_matched_obs_map_test.png", dpi=150)
    plt.close(fig)
    print(f"  → Saved: {OUT_DIR}/rides00_matched_obs_map_test.png")


# ─────────────────────────────────────────────────────────────────────────────
# E) matched obs+map SEQUENCE — 한 episode 안에서 시간에 따라 obs/map/예측이
#    어떻게 같이 변하는지 연속적으로 확인 (도로 형태를 실제로 따라가는지 검증)
# ─────────────────────────────────────────────────────────────────────────────

def run_matched_sequence_test(model, n_steps=6):
    print("\n" + "=" * 60)
    print("E) MATCHED OBS+MAP SEQUENCE TEST (한 episode 내 시간 흐름)")
    print("=" * 60)

    stats_list = classify_rides00_segments()
    straight_seg = sorted([s for s in stats_list if s["scenario"] == "straight"],
                           key=lambda s: s["curvature_density"])
    curve_seg = sorted([s for s in stats_list if s["scenario"] in
                        ("curve_left", "curve_right", "sharp_left_turn", "sharp_right_turn")],
                       key=lambda s: -s["curvature_density"])

    targets = []
    if straight_seg:
        targets.append(("straight", straight_seg[0]))
    if curve_seg:
        targets.append(("curve", curve_seg[0]))

    for label, seg_stat in targets:
        _run_one_sequence(model, label, seg_stat, n_steps)


def _run_one_sequence(model, label, seg_stat, n_steps):
    import csv
    ep = int(seg_stat["seg_key"].split("_")[1])
    raw = _load_raw_episode_table(ep)
    if raw is None:
        print(f"  ⚠ {seg_stat['seg_key']}: raw episode {ep} not found, skip")
        return

    with open(seg_stat["seg_dir"] / "gps.csv") as f:
        rows = list(csv.DictReader(f))
    seg_lat = np.array([float(r["latitude"]) for r in rows])
    seg_lon = np.array([float(r["longitude"]) for r in rows])
    offset = _find_segment_offset(seg_lat, seg_lon, raw["lat"], raw["lon"])
    print(f"[{label}] {seg_stat['seg_key']}  scenario={seg_stat['scenario']}  "
          f"curv_density={seg_stat['curvature_density']:.1f} deg/100m  offset={offset}")

    obs_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])
    video_reader = VideoReader()
    CTX_STRIDE, N_CTX = 3, 5
    PAST_MARGIN = CTX_STRIDE * N_CTX

    n_pngs = len(list(seg_stat["seg_dir"].glob("osm_map_*.png")))
    fracs = np.linspace(0.15, 0.85, n_steps)
    steps = []

    for frac in fracs:
        local_idx = min(int(len(rows) * frac), len(rows) - 1, n_pngs - 1)
        global_row = offset + local_idx
        global_row = max(PAST_MARGIN, min(global_row, len(raw["frame_index"]) - 1))

        ctx_rows = [max(0, global_row - CTX_STRIDE * (N_CTX - k)) for k in range(N_CTX)] + [global_row]
        imgs = [obs_transform(video_reader.get_frame(raw["video_path"], float(raw["timestamp"][r])))
                for r in ctx_rows]
        obs_stack = torch.cat(imgs, dim=0).unsqueeze(0).to(DEVICE)
        obs_cur = obs_stack[:, -3:]

        map_png = (seg_stat["seg_dir"] / f"osm_map_{local_idx:06d}.png")
        if not map_png.exists():
            pngs = sorted(seg_stat["seg_dir"].glob("osm_map_*.png"))
            map_png = pngs[min(local_idx, len(pngs) - 1)]
        map_tensor = load_map_tensor(map_png, DEVICE)

        goal_pose = torch.zeros(1, 4, device=DEVICE)
        goal_mask = torch.zeros(1, dtype=torch.long, device=DEVICE)
        feat_text = torch.zeros(1, 512, device=DEVICE)
        cur_img = F.interpolate(obs_cur, (224, 224), mode='bilinear', align_corners=False)

        with torch.no_grad():
            pred, _, _ = model(obs_stack, goal_pose, map_tensor, obs_cur, goal_mask, feat_text, cur_img)
        pred_m = pred[0, :, :2].detach().cpu().numpy() * METRIC_WAYPOINT_SPACING

        steps.append({
            "frac": frac, "local_idx": local_idx,
            "cam_img": denorm_img(obs_stack[0, -3:]),
            "map_img": denorm_img(map_tensor[0]),
            "pred_m": pred_m,
        })
        print(f"    frac={frac:.2f} local_idx={local_idx}  lateral(sum)={pred_m[:,1].sum():+.3f}m")

    video_reader.close_all()
    if not steps:
        return

    # 공통 축 범위
    all_x, all_y = [], []
    for s in steps:
        px, py = to_ego_plot_xy(s["pred_m"])
        all_x.append(px); all_y.append(py)
    all_x = np.concatenate(all_x); all_y = np.concatenate(all_y)
    xmax = max(0.4, np.abs(all_x).max() * 1.25)
    ymin = min(0.0, all_y.min()) - 0.15
    ymax = all_y.max() * 1.15 + 0.15

    n = len(steps)
    fig, axes = plt.subplots(3, n, figsize=(2.6 * n, 8.5))
    for col, s in enumerate(steps):
        axes[0, col].imshow(s["cam_img"])
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
        axes[0, col].set_title(f"t={s['frac']:.2f}\n(local_idx={s['local_idx']})", fontsize=8)
        if col == 0:
            axes[0, col].set_ylabel("camera", fontsize=9)

        ax = axes[1, col]
        px, py = to_ego_plot_xy(s["pred_m"])
        ax.plot(px, py, '-o', ms=4, lw=2, color="#1E88E5" if label == "straight" else "#E53935")
        ax.plot(0, 0, 'k*', ms=9, zorder=5)
        ax.axhline(0, color='lightgray', lw=0.6); ax.axvline(0, color='lightgray', lw=0.6)
        ax.set_xlim(-xmax, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=6)
        if col == 0:
            ax.set_ylabel("predicted traj\n(ego frame)", fontsize=9)

        axes[2, col].imshow(s["map_img"])
        axes[2, col].set_xticks([]); axes[2, col].set_yticks([])
        if col == 0:
            axes[2, col].set_ylabel("odom map", fontsize=9)

    fig.suptitle(f"Matched obs+map sequence — {label} segment ({seg_stat['seg_key']}, "
                 f"scenario={seg_stat['scenario']}, curv_density={seg_stat['curvature_density']:.0f} deg/100m)\n"
                 f"실제 카메라+map이 시간에 따라 함께 변할 때 예측 궤적의 변화", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = OUT_DIR / f"rides00_sequence_{label}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(CKPT))
    parser.add_argument("--method", type=str, default="all",
                        choices=["heldout", "swap", "matched", "sequence", "all"])
    args = parser.parse_args()

    model = load_model(args.ckpt)

    if args.method in ("heldout", "all"):
        run_heldout_ablation(model)

    if args.method in ("swap", "all"):
        run_map_swap_test(model)

    if args.method in ("matched", "all"):
        run_matched_obs_map_test(model)

    if args.method in ("sequence", "all"):
        run_matched_sequence_test(model)

    print(f"\n모든 결과 저장 위치: {OUT_DIR}")


if __name__ == "__main__":
    main()
