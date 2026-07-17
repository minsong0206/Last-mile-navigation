"""
vis_seg02_epochs.py

ep0405 seg02의 모든 valid 프레임에 대해 epoch 5/10/15/20 체크포인트 예측을
GT trajectory와 함께 오버레이한 대형 PNG 이미지를 생성한다.

출력 디렉토리:
  checkpoints/omnivla_edge_rides11/vis/ep0405_seg02/frame_{fi:06d}.png

레이아웃 (1행 × 3열):
  Col 1: 현재 카메라 이미지 (raw, 정규화 역변환)
  Col 2: Trajectory plot — GT(녹색) + ep5/10/15/20 예측 오버레이
  Col 3: OSM 맵 오버레이 (원본 224×224 PNG 사용, PX_PER_M=4.48)
          — GT(녹색 점선) + ep5/10/15/20 예측 (색상 구분)

OSM 맵 스케일 수정:
  기존 코드는 96×96 리사이즈된 맵을 사용 → PX_PER_M=1.92 (waypoint가 너무 작음)
  이 스크립트는 원본 224×224 PNG를 직접 로드 → PX_PER_M=4.48 (2.3× 더 큰 표시)

실행:
  conda run -n mbra python vis_seg02_epochs.py [--device cuda] [--batch 16]
"""

import os
import sys
import math
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pyarrow as pa

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE       = Path("/media/ms/WD_BLACK_4TB")
ARROW_PATH = BASE / "Learning-to-Drive-Anywhere-with-MBRA/FrodoBots-2K/processed/output_rides_11/train/data-00000-of-00001.arrow"
SCORES_PATH= BASE / "Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/osm_data/output_rides_11/episode_scores.json"
OSM_ROOT   = BASE / "Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/osm_data/output_rides_11/osm_maps_arrow"
FRAME_ROOT = BASE / "Learning-to-Drive-Anywhere-with-MBRA/FrodoBots-2K/processed/output_rides_11/frames"
CKPT_DIR   = BASE / "Learning-to-Drive-Anywhere-with-MBRA/checkpoints/omnivla_edge_rides11"
OUT_DIR    = CKPT_DIR / "vis" / "ep0405_seg02"

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "third_party" / "omnivla" / "inference"))
sys.path.insert(0, str(BASE / "Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/py"))

from model_omnivla_edge import OmniVLA_edge
from episode_selector import split_into_segments

# ── 상수 ──────────────────────────────────────────────────────────────────────
METRIC_WAYPOINT_SPACING = 0.125
N_WAYPOINTS   = 8
WAYPOINT_STRIDE = 3
CTX_STRIDE    = 3
N_CTX         = 5
PAST_MARGIN   = CTX_STRIDE * N_CTX     # 15
FUTURE_MARGIN = WAYPOINT_STRIDE * N_WAYPOINTS  # 24

IMG_SIZE_OBS = (96, 96)
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]
MAP_RANGE_M = 25.0   # 맵 반폭 (m) — osm_map_generator.py와 동일
MAP_SIZE_PX = 224    # 원본 OSM PNG 크기

MODEL_PARAMS = {
    "context_size":            5,
    "len_traj_pred":           8,
    "learn_angle":             True,
    "obs_encoder":             "efficientnet-b0",
    "obs_encoding_size":       1024,
    "late_fusion":             False,
    "mha_num_attention_heads": 4,
    "mha_num_attention_layers":4,
    "mha_ff_dim_factor":       4,
}

# 체크포인트 목록: (epoch 번호, 파일명, 표시 색상, 선 스타일)
CHECKPOINTS = [
    (5,  "epoch_005.pth", "#64B5F6", "-o"),   # 하늘색
    (10, "epoch_010.pth", "#FFA726", "-o"),   # 주황
    (15, "epoch_015.pth", "#EF5350", "-o"),   # 빨강
    (20, "epoch_020.pth", "#CE93D8", "-o"),   # 보라
]

obs_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE_OBS),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD),
])
map_transform_for_tensor = transforms.Compose([
    transforms.Resize(IMG_SIZE_OBS),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD),
])


# ══════════════════════════════════════════════════════════════════════════════
# 모델 로드
# ══════════════════════════════════════════════════════════════════════════════

def build_model(device: torch.device) -> OmniVLA_edge:
    model = OmniVLA_edge(
        context_size=MODEL_PARAMS["context_size"],
        len_traj_pred=MODEL_PARAMS["len_traj_pred"],
        learn_angle=MODEL_PARAMS["learn_angle"],
        obs_encoder=MODEL_PARAMS["obs_encoder"],
        obs_encoding_size=MODEL_PARAMS["obs_encoding_size"],
        late_fusion=MODEL_PARAMS["late_fusion"],
        mha_num_attention_heads=MODEL_PARAMS["mha_num_attention_heads"],
        mha_num_attention_layers=MODEL_PARAMS["mha_num_attention_layers"],
        mha_ff_dim_factor=MODEL_PARAMS["mha_ff_dim_factor"],
    )
    return model.to(device).eval()


def load_checkpoint(model: OmniVLA_edge, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 데이터 준비: ep0405 seg02 valid frame list 구성
# ══════════════════════════════════════════════════════════════════════════════

def load_arrow():
    print(f"Loading Arrow: {ARROW_PATH}")
    table = pa.ipc.open_stream(open(str(ARROW_PATH), "rb")).read_all()
    ep_arr = np.array(table["episode_index"].to_pylist(), dtype=np.int64)
    fi_arr = np.array(table["frame_index"].to_pylist(),   dtype=np.int64)
    fp_arr = np.array(table["observation.filtered_position"].to_pylist(), dtype=np.float32)
    fh_arr = np.array(table["observation.filtered_heading"].to_pylist(),  dtype=np.float32)
    lat_arr= np.array(table["observation.latitude"].to_pylist(),  dtype=np.float64)
    lon_arr= np.array(table["observation.longitude"].to_pylist(), dtype=np.float64)
    return ep_arr, fi_arr, fp_arr, fh_arr, lat_arr, lon_arr


def build_seg02_samples(
    ep_arr, fi_arr, fp_arr, fh_arr, lat_arr, lon_arr,
    ep: int = 405, seg_idx: int = 2,
) -> Tuple[List, Dict, np.ndarray, np.ndarray]:
    """
    ep0405 seg02의 valid frame list를 반환.
    Returns:
      valid_frames: list of (fi, seg_local_idx)
      global_row_map: dict[(ep, fi)] → global row index
      seg_fp:  (N, 2) — segment EKF positions
      seg_fh:  (N,)   — segment headings
    """
    mask = ep_arr == ep
    ep_fi  = fi_arr[mask]
    ep_fp  = fp_arr[mask]
    ep_fh  = fh_arr[mask]
    ep_lat = lat_arr[mask]
    ep_lon = lon_arr[mask]

    # global_row_map for waypoint computation
    global_row_map = {}
    for i, (ep_v, fi_v) in enumerate(zip(ep_arr, fi_arr)):
        global_row_map[(int(ep_v), int(fi_v))] = i

    segments = split_into_segments(ep_fp, ep_lat, ep_lon)
    if seg_idx >= len(segments):
        raise ValueError(f"ep{ep:04d} has only {len(segments)} segments")

    seg = segments[seg_idx]
    idxs = seg["frame_indices"]      # episode-local indices
    seg_fi = ep_fi[idxs]             # actual frame_index values
    fi_start = int(seg_fi[0])
    fi_end   = int(seg_fi[-1])

    print(f"ep{ep:04d} seg{seg_idx:02d}: {len(idxs)} frames, fi={fi_start}~{fi_end}")

    valid_frames = []
    for local_idx, fi in enumerate(seg_fi.tolist()):
        fi = int(fi)
        if fi < fi_start + PAST_MARGIN:   continue
        if fi > fi_end   - FUTURE_MARGIN: continue
        valid_frames.append((fi, local_idx))

    print(f"  valid frames: {len(valid_frames)}, "
          f"fi={valid_frames[0][0]}~{valid_frames[-1][0]}")
    return valid_frames, global_row_map, fp_arr, fh_arr


# ══════════════════════════════════════════════════════════════════════════════
# 단일 프레임 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════

def load_frame_data(
    ep: int,
    fi: int,
    seg_local_idx: int,
    global_row_map: Dict,
    fp_arr: np.ndarray,
    fh_arr: np.ndarray,
    seg_idx: int = 2,
) -> Dict:
    """
    한 프레임의 모든 입력 데이터를 로드.
    반환:
      raw_img:      PIL Image (카메라 원본, 정규화 없음)
      raw_osm:      PIL Image (OSM 224×224, 정규화 없음)
      obs_stack_t:  (18, 96, 96) tensor (정규화됨)
      map_tile_t:   (9,  96, 96) tensor (정규화됨)
      gt_wp:        (8, 2) numpy — 실제 미터값 (METRIC_WAYPOINT_SPACING 곱한 후)
    """
    frame_dir = FRAME_ROOT / f"episode_{ep:04d}"

    # ── 카메라 이미지 스택 ──────────────────────────────────────────────────
    ctx_tensors = []
    for k in range(N_CTX):
        fi_ctx = fi - CTX_STRIDE * (N_CTX - k)
        path = frame_dir / f"{fi_ctx:06d}.jpg"
        img = Image.open(str(path)).convert("RGB")
        ctx_tensors.append(obs_transform(img))

    raw_img = Image.open(str(frame_dir / f"{fi:06d}.jpg")).convert("RGB")
    obs_t   = obs_transform(raw_img)  # (3, 96, 96)
    obs_stack_t = torch.cat(ctx_tensors + [obs_t], dim=0)  # (18, 96, 96)

    # ── OSM 맵 ──────────────────────────────────────────────────────────────
    osm_path = OSM_ROOT / f"episode_{ep:04d}_seg{seg_idx:02d}" / f"osm_map_{seg_local_idx:06d}.png"
    raw_osm  = Image.open(str(osm_path)).convert("RGB")
    map_tile_t_96  = map_transform_for_tensor(raw_osm)         # (3, 96, 96)
    map_images_t   = torch.cat([map_tile_t_96, map_tile_t_96, obs_t], dim=0)  # (9, 96, 96)

    # ── GT waypoints (실제 미터) ─────────────────────────────────────────────
    row_curr = global_row_map[(ep, fi)]
    pos_curr = fp_arr[row_curr]
    hdg_curr = float(fh_arr[row_curr])
    cos_h, sin_h = math.cos(hdg_curr), math.sin(hdg_curr)

    waypoints_m = []
    for k in range(1, N_WAYPOINTS + 1):
        fi_fut = fi + k * WAYPOINT_STRIDE
        row_fut = global_row_map.get((ep, fi_fut))
        if row_fut is None:
            waypoints_m.append(waypoints_m[-1] if waypoints_m else [0.0, 0.0])
            continue
        pos_fut = fp_arr[row_fut]
        dx = pos_fut[0] - pos_curr[0]
        dy = pos_fut[1] - pos_curr[1]
        x_ego =  dx * cos_h + dy * sin_h
        y_ego = -dx * sin_h + dy * cos_h
        waypoints_m.append([x_ego, y_ego])

    gt_wp_m = np.array(waypoints_m, dtype=np.float32)  # (8, 2) 실제 미터

    return {
        "raw_img":     raw_img,
        "raw_osm":     raw_osm,
        "obs_stack_t": obs_stack_t,
        "map_images_t":map_images_t,
        "gt_wp_m":     gt_wp_m,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 추론 (배치 단위)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference_batch(
    model: OmniVLA_edge,
    obs_stack_list:  List[torch.Tensor],   # (N,) × (18,96,96)
    map_images_list: List[torch.Tensor],   # (N,) × (9,96,96)
    device: torch.device,
) -> np.ndarray:
    """
    N개 프레임을 배치로 추론.
    반환: (N, 8, 2) 실제 미터 단위 예측 waypoints
    """
    B = len(obs_stack_list)
    obs_stack  = torch.stack(obs_stack_list).to(device)    # (B, 18, 96, 96)
    map_images = torch.stack(map_images_list).to(device)   # (B, 9,  96, 96)
    obs_cur    = obs_stack[:, -3:, :, :]                   # (B, 3,  96, 96)

    goal_pose  = torch.zeros(B, 4,   device=device)
    goal_img   = obs_cur                                   # (B, 3,  96, 96)
    goal_mask  = torch.zeros(B,      device=device, dtype=torch.long)
    feat_text  = torch.zeros(B, 512, device=device)
    current_img= F.interpolate(obs_cur, (224, 224), mode="bilinear", align_corners=False)

    action_pred, _, _ = model(
        obs_stack, goal_pose, map_images,
        goal_img, goal_mask, feat_text, current_img,
    )

    pred_norm = action_pred[:, :, :2].cpu().numpy()  # (B, 8, 2) 정규화됨
    pred_m    = pred_norm * METRIC_WAYPOINT_SPACING  # 실제 미터
    return pred_m


# ══════════════════════════════════════════════════════════════════════════════
# 시각화: 1프레임 → 1 PNG
# ══════════════════════════════════════════════════════════════════════════════

def make_figure(
    fi: int,
    seg_local_idx: int,
    raw_img:   Image.Image,
    raw_osm:   Image.Image,
    gt_wp_m:   np.ndarray,            # (8, 2) 미터
    preds_m:   Dict[str, np.ndarray], # epoch_label → (8, 2) 미터
    fixed_xlim: tuple = None,         # 전역 고정 x 범위 (-xabs, xabs)
    fixed_ylim: tuple = None,         # 전역 고정 y 범위 (ymin, ymax)
) -> plt.Figure:
    """
    3-패널 대형 PNG (V1 기준).
    Col 1: 카메라 이미지
    Col 2: Trajectory plot — 데이터 범위 자동 스케일 + aspect='equal'
    Col 3: OSM 맵 오버레이 (원본 224×224 전체)
    """
    epoch_styles = {
        "Epoch 5":  ("#64B5F6", 2.5),
        "Epoch 10": ("#FFA726", 2.5),
        "Epoch 15": ("#EF5350", 2.8),
        "Epoch 20": ("#CE93D8", 3.0),
    }

    fig, axes = plt.subplots(1, 3, figsize=(28, 10),
                             gridspec_kw={"width_ratios": [1, 3, 1]})
    fig.patch.set_facecolor("#1a1a2e")

    # ── Col 1: 카메라 이미지 ─────────────────────────────────────────────────
    ax_cam = axes[0]
    ax_cam.imshow(raw_img)
    ax_cam.set_title(
        f"ep0405_seg02  fi={fi:06d}  seg_local={seg_local_idx:06d}\nCamera (current frame)",
        fontsize=11, color="white", pad=8,
    )
    ax_cam.axis("off")

    # ── Col 2: Trajectory plot ──────────────────────────────────────────────
    ax_tr = axes[1]
    ax_tr.set_facecolor("#0d0d1a")

    gt_x = -gt_wp_m[:, 1]
    gt_y =  gt_wp_m[:, 0]
    ax_tr.plot(
        np.insert(gt_x, 0, 0.0), np.insert(gt_y, 0, 0.0),
        "--o", color="#00E676", linewidth=3.5, markersize=12,
        markeredgecolor="white", markeredgewidth=0.8, label="GT", zorder=10,
    )

    for label, pred_m in preds_m.items():
        color, lw = epoch_styles[label]
        px = -pred_m[:, 1]
        py =  pred_m[:, 0]
        ax_tr.plot(
            np.insert(px, 0, 0.0), np.insert(py, 0, 0.0),
            "-o", color=color, linewidth=lw, markersize=9,
            markeredgecolor="black", markeredgewidth=0.4,
            label=label, alpha=0.9, zorder=8,
        )

    ax_tr.plot(0, 0, "*", color="cyan", markersize=22,
               markeredgecolor="black", markeredgewidth=1.0, zorder=15)

    # 전역 고정 limits 사용 (없으면 이 프레임 데이터 기반 fallback)
    if fixed_xlim is not None and fixed_ylim is not None:
        ax_tr.set_xlim(fixed_xlim)
        ax_tr.set_ylim(fixed_ylim)
    else:
        all_x = np.concatenate([gt_x] + [-p[:, 1] for p in preds_m.values()] + [[0.0]])
        all_y = np.concatenate([gt_y] + [ p[:, 0] for p in preds_m.values()] + [[0.0]])
        pad_x = max(abs(all_x).max() * 0.3, 0.3)
        pad_y = max(abs(all_y).max() * 0.3, 0.5)
        xabs  = max(abs(all_x.min() - pad_x), abs(all_x.max() + pad_x))
        ax_tr.set_xlim(-xabs, xabs)
        ax_tr.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)
    ax_tr.set_aspect("equal", adjustable="box")

    ax_tr.set_xlabel("Left / Right (m)", fontsize=14, color="white")
    ax_tr.set_ylabel("Forward (m)",      fontsize=14, color="white")
    ax_tr.set_title("Trajectory: GT vs Epoch Predictions", fontsize=15,
                    color="white", pad=10)
    ax_tr.legend(fontsize=13, facecolor="#1a1a2e", edgecolor="gray",
                 labelcolor="white", loc="upper left")
    ax_tr.grid(True, alpha=0.25, color="white", linestyle="--")
    ax_tr.tick_params(colors="white", labelsize=12)
    for spine in ax_tr.spines.values():
        spine.set_edgecolor("#555555")

    # ── Col 3: OSM 맵 오버레이 (224×224 전체) ───────────────────────────────
    ax_osm = axes[2]
    osm_np = np.array(raw_osm)
    ax_osm.imshow(osm_np)

    H, W  = osm_np.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    PX_PER_M = H / (MAP_RANGE_M * 2.0)

    gt_px = cx - gt_wp_m[:, 1] * PX_PER_M
    gt_py = cy - gt_wp_m[:, 0] * PX_PER_M
    ax_osm.plot(
        np.insert(gt_px, 0, cx), np.insert(gt_py, 0, cy),
        "--o", color="#00E676", linewidth=2.5, markersize=8,
        markeredgecolor="black", markeredgewidth=0.5, label="GT", zorder=10,
    )

    for label, pred_m in preds_m.items():
        color, lw = epoch_styles[label]
        pred_px = cx - pred_m[:, 1] * PX_PER_M
        pred_py = cy - pred_m[:, 0] * PX_PER_M
        ax_osm.plot(
            np.insert(pred_px, 0, cx), np.insert(pred_py, 0, cy),
            "-o", color=color, linewidth=lw, markersize=6,
            markeredgecolor="black", markeredgewidth=0.3,
            label=label, alpha=0.9, zorder=8,
        )

    ax_osm.plot(cx, cy, "*", color="cyan", markersize=18,
                markeredgecolor="black", markeredgewidth=0.8, zorder=15)

    ax_osm.set_title("OSM Map Overlay (224×224 full)", fontsize=11,
                     color="white", pad=8)
    ax_osm.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="gray",
                  labelcolor="white", loc="lower right")
    ax_osm.axis("off")

    fig.tight_layout(pad=1.5)
    fig.suptitle(
        f"OmniVLA-Edge Fine-tuning  |  ep0405_seg02  fi={fi:06d}  "
        f"(seg_local={seg_local_idx:06d})",
        fontsize=13, color="white", y=1.01,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Arrow 로드 ──────────────────────────────────────────────────────────
    ep_arr, fi_arr, fp_arr, fh_arr, lat_arr, lon_arr = load_arrow()

    # ── ep0405 seg02 valid frame list ───────────────────────────────────────
    valid_frames, global_row_map, fp_all, fh_all = build_seg02_samples(
        ep_arr, fi_arr, fp_arr, fh_arr, lat_arr, lon_arr,
        ep=405, seg_idx=2,
    )
    print(f"Total valid frames to visualize: {len(valid_frames)}")

    # ── test 모드: 첫 프레임 1장만, vis/ 루트에 저장 ────────────────────────
    if args.test:
        valid_frames = valid_frames[:1]

    BATCH = args.batch

    # ── 1단계: GT waypoints + 텐서 사전 수집 (첫 checkpoint에서만) ────────────
    print("\n── 데이터 사전 수집 ──")
    gt_all:  List[np.ndarray] = []   # (N,) × (8,2) 미터
    obs_cache: List[torch.Tensor] = []
    map_cache: List[torch.Tensor] = []

    for fi, seg_local_idx in tqdm(valid_frames, desc="  Loading frames"):
        d = load_frame_data(405, fi, seg_local_idx, global_row_map,
                            fp_all, fh_all, seg_idx=2)
        gt_all.append(d["gt_wp_m"])
        obs_cache.append(d["obs_stack_t"])
        map_cache.append(d["map_images_t"])

    # ── 2단계: 각 checkpoint 추론 ────────────────────────────────────────────
    preds_all: Dict[str, List[np.ndarray]] = {}

    print("\n── 모델 추론 단계 ──")
    for epoch_num, ckpt_fname, color, _ in CHECKPOINTS:
        label     = f"Epoch {epoch_num}"
        ckpt_path = CKPT_DIR / ckpt_fname
        print(f"  Loading {ckpt_fname} ...")

        model = build_model(device)
        model = load_checkpoint(model, ckpt_path, device)

        preds_this: List[np.ndarray] = []
        n_batches = math.ceil(len(valid_frames) / BATCH)

        for b in tqdm(range(n_batches), desc=f"  Ep{epoch_num} inference"):
            obs_list = obs_cache[b * BATCH : (b + 1) * BATCH]
            map_list = map_cache[b * BATCH : (b + 1) * BATCH]
            pred_m   = run_inference_batch(model, obs_list, map_list, device)
            for i in range(len(obs_list)):
                preds_this.append(pred_m[i])

        preds_all[label] = preds_this
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 텐서 캐시 해제
    del obs_cache, map_cache

    # ── 전역 trajectory limits 계산 (모든 프레임에서 일관된 그래프 크기) ──────
    all_x_global, all_y_global = [], []
    for i, gt in enumerate(gt_all):
        all_x_global.append(-gt[:, 1])
        all_y_global.append( gt[:, 0])
        for label in preds_all:
            p = preds_all[label][i]
            all_x_global.append(-p[:, 1])
            all_y_global.append( p[:, 0])
    all_x_global = np.concatenate(all_x_global + [[0.0]])
    all_y_global = np.concatenate(all_y_global + [[0.0]])
    pad_x = max(abs(all_x_global).max() * 0.3, 0.3)
    pad_y = max(abs(all_y_global).max() * 0.3, 0.5)
    g_xabs = max(abs(all_x_global.min() - pad_x), abs(all_x_global.max() + pad_x))
    g_ymin = all_y_global.min() - pad_y
    g_ymax = all_y_global.max() + pad_y
    fixed_xlim = (-g_xabs, g_xabs)
    fixed_ylim = (g_ymin,  g_ymax)
    print(f"  Global trajectory limits: x={fixed_xlim}, y={fixed_ylim}")

    # ── 3단계: 시각화 & 저장 ─────────────────────────────────────────────────
    print("\n── 시각화 저장 단계 ──")
    frame_dir = FRAME_ROOT / "episode_0405"

    for idx, (fi, seg_local_idx) in enumerate(
        tqdm(valid_frames, desc="  Saving PNGs")
    ):
        if args.test:
            out_path = CKPT_DIR / "vis" / f"test_ep0405_seg02_fi{fi:06d}.png"
        else:
            out_path = OUT_DIR / f"frame_{fi:06d}.png"
        if out_path.exists() and not args.overwrite:
            continue

        raw_img = Image.open(str(frame_dir / f"{fi:06d}.jpg")).convert("RGB")
        raw_osm = Image.open(str(
            OSM_ROOT / "episode_0405_seg02" / f"osm_map_{seg_local_idx:06d}.png"
        )).convert("RGB")

        preds_m = {label: preds_all[label][idx] for label in preds_all}
        gt_wp_m = gt_all[idx]

        fig = make_figure(fi, seg_local_idx, raw_img, raw_osm, gt_wp_m, preds_m,
                          fixed_xlim=fixed_xlim, fixed_ylim=fixed_ylim)
        fig.savefig(str(out_path), dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

    print(f"\nDone. {len(valid_frames)} PNGs saved → {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",    type=str,  default="cuda:0")
    parser.add_argument("--batch",     type=int,  default=16,
                        help="추론 배치 크기")
    parser.add_argument("--overwrite", action="store_true",
                        help="기존 PNG 덮어쓰기")
    parser.add_argument("--test", action="store_true",
                        help="첫 프레임 1장만 생성 → vis/test_*.png")
    main(parser.parse_args())
