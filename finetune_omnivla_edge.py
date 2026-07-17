"""
finetune_omnivla_edge.py

OmniVLA-Edge-Odom을 FrodoBots rides_11 데이터셋으로 fine-tuning하는 스크립트.

변환된 체크포인트(omnivla-edge-odom3ch.pth)를 로드하고,
rides11_dataset.py의 Rides11Dataset으로 학습한다.
goal_encoder는 OSM 맵 1장(3ch)만 입력받도록 수정된 OmniVLA_edge_odom 사용.

실행:
  cd /media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA
  conda run -n mbra python finetune_omnivla_edge.py --config config/rides11_finetune.yaml

평가 전용 (test set trajectory 시각화):
  conda run -n mbra python finetune_omnivla_edge.py --eval_only \
    --eval_ckpt checkpoints/omnivla_edge_rides11_odom/best.pth

모델 입력 (forward 시그니처):
  obs_img     (B, 18, 96, 96)   — 과거 5프레임 + 현재 1프레임 (각 3ch, concat)
  goal_pose   (B, 4)            — [x, y, cos, sin] ego-frame 목표 위치 (rides_11: zeros)
  map_images  (B, 3, 96, 96)    — OSM 맵 1장 (odom3ch 모델)
  goal_img    (B, 3, 96, 96)    — obs현재 (모델 내부에서 6ch으로 cat)
  goal_mask   (B,)              — modality_id: 0=satellite only (language/pose 없음)
  feat_text   (B, 512)          — CLIP 언어 피처 (rides_11: zeros)
  current_img (B, 3, 224, 224)  — FiLM 컨디셔닝용 고해상도 현재 이미지

모델 출력:
  action_pred (B, 8, 4)         — 8-step waypoints [x, y, cos_h, sin_h]
  dist_pred   (B, 1)            — 거리 예측 (학습에 사용 안 함)
  no_goal_mask (B,)             — 내부 mask ID (무시)

Loss:
  L_wp     = smooth_l1_loss(action_pred[:,:,:2], gt_waypoints)
  L_smooth = acceleration penalty (연속 waypoint 간 가속도 패널티)
  L_total  = L_wp + 0.1 * L_smooth

평가 지표:
  ADE (Average Displacement Error): 8개 waypoint 평균 거리 오차 (미터)
  FDE (Final Displacement Error):   마지막 waypoint(2.4초 후) 거리 오차 (미터)
  → 정규화된 예측값 × METRIC_WAYPOINT_SPACING(0.125m)로 실제 미터로 환산
"""

import os
import sys
import math
import yaml
import argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
import matplotlib
matplotlib.use("Agg")  # 화면 없는 서버 환경에서 matplotlib 사용 (display 불필요)
import matplotlib.pyplot as plt

import clip

# OmniVLA-Edge-Odom — goal_encoder in_channels=3 (OSM 맵 1장)
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "third_party" / "omnivla" / "inference"))

from model_omnivla_edge_odom import OmniVLA_edge_odom

# rides_11 데이터셋 — Arrow + episode_scores.json + OSM 맵 이미지
sys.path.insert(0, "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/py")
from rides11_dataset import Rides11Dataset, METRIC_WAYPOINT_SPACING

# ── 기본 경로 설정 ─────────────────────────────────────────────────────────────
BASE = Path("/media/ms/WD_BLACK_4TB")

# Step 1에서 변환된 체크포인트 (goal_encoder 9ch→3ch 평균화)
CKPT_PATH   = BASE / "OmniVLA/omnivla-edge/omnivla-edge-odom3ch.pth"

# rides_11 Arrow 데이터 (convert_to_hf.py로 생성)
ARROW_PATH  = BASE / "Learning-to-Drive-Anywhere-with-MBRA/FrodoBots-2K/processed/output_rides_11/train/data-00000-of-00001.arrow"

# episode_selector.py 결과 (선별된 세그먼트 목록)
SCORES_PATH = BASE / "Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/osm_data/output_rides_11/episode_scores.json"

# osm_map_generator.py 결과 (Arrow frame_index 기준 224×224 PNG)
OSM_ROOT    = BASE / "Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/osm_data/output_rides_11/osm_maps_arrow"

# Arrow의 videos/ 경로 기준 루트 (videos/ride_XXXX.mp4 → 이 디렉토리 아래)
VIDEO_ROOT  = BASE / "Learning-to-Drive-Anywhere-with-MBRA/FrodoBots-2K/processed/output_rides_11"

# 체크포인트 저장 디렉토리
SAVE_DIR    = BASE / "Learning-to-Drive-Anywhere-with-MBRA/checkpoints/omnivla_edge_rides11_odom"


# ── 모델 파라미터 (inference/run_omnivla_edge.py와 동일) ─────────────────────
# 저자 체크포인트와 아키텍처가 정확히 일치해야 strict=True 로드 가능
MODEL_PARAMS = {
    "model_type":              "omnivla-edge",
    "context_size":            5,       # 과거 프레임 수 (M=5, 논문 Fig.7)
    "len_traj_pred":           8,       # 예측 waypoint 수
    "learn_angle":             True,    # action이 [x, y, cos, sin] 4차원
    "obs_encoder":             "efficientnet-b0",  # 관측 인코더 (pretrained)
    "obs_encoding_size":       1024,    # EfficientNet 출력 임베딩 크기
    "late_fusion":             False,   # goal image를 obs와 early fusion
    "mha_num_attention_heads": 4,       # Transformer multi-head attention 헤드 수
    "mha_num_attention_layers": 4,      # Transformer 레이어 수
    "mha_ff_dim_factor":       4,       # FFN hidden dim = embed_dim * factor
    "clip_type":               "ViT-B/32",  # CLIP 텍스트 인코더 종류
}

# ── 학습 하이퍼파라미터 기본값 ───────────────────────────────────────────────
TRAIN_CONFIG = {
    "epochs":        20,
    "batch_size":    32,
    "lr":            1e-4,
    "weight_decay":  1e-4,
    "val_ratio":     0.1,
    "test_ratio":    0.1,
    "num_workers":   4,
    "smooth_weight": 0.1,
    "save_freq":     5,
    "log_freq":      50,
    "vis_freq":      1,       # 몇 epoch마다 waypoint 시각화 이미지를 wandb에 업로드
    "freeze":        "partial",
    "use_wandb":     True,
    "wandb_project": "omnivla-edge-rides11-odom",
}

# modality_id=0: "satellite only" (OSM map만 사용, pose/goal_image/language 없음)
# inference/run_omnivla_edge.py의 modality_id 결정 로직 참고
MODALITY_ID = 0


# ══════════════════════════════════════════════════════════════════════════════
# 모델 로드
# ══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path: str, model_params: dict, device: torch.device):
    """
    저자 체크포인트에서 OmniVLA_edge 모델과 CLIP 텍스트 인코더를 로드.
    inference/utils_policy.py의 load_model()과 동일한 로직.
    체크포인트는 state_dict 형태로 저장되어 있음 (DDP wrapper 없음).
    """
    model = OmniVLA_edge_odom(
        context_size=model_params["context_size"],
        len_traj_pred=model_params["len_traj_pred"],
        learn_angle=model_params["learn_angle"],
        obs_encoder=model_params["obs_encoder"],
        obs_encoding_size=model_params["obs_encoding_size"],
        late_fusion=model_params["late_fusion"],
        mha_num_attention_heads=model_params["mha_num_attention_heads"],
        mha_num_attention_layers=model_params["mha_num_attention_layers"],
        mha_ff_dim_factor=model_params["mha_ff_dim_factor"],
    )

    # CLIP 텍스트 인코더 — rides_11에서는 zeros 입력이므로 eval()로만 사용
    text_encoder, _ = clip.load(model_params["clip_type"])
    text_encoder = text_encoder.to(torch.float32)

    print(f"[load_model] Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    print("[load_model] Checkpoint loaded successfully.")

    return model, text_encoder


# ══════════════════════════════════════════════════════════════════════════════
# Freeze 전략
# ══════════════════════════════════════════════════════════════════════════════

def apply_freeze(model: nn.Module, strategy: str):
    """
    Freeze 전략에 따라 모델 파라미터의 requires_grad를 설정.

    "map_encoder_only": OSM map encoder(goal_encoder, 9ch EfficientNet)만 학습.
                        나머지 freeze. Stage 1 — 빠른 초기 적응.

    "partial":          obs_encoder(관측 EfficientNet-B0) freeze.
                        transformer(decoder) + action_predictor 학습. Stage 2 — 권장.

    "none":             전체 파라미터 학습. Stage 3 — 과적합 위험.
    """
    if strategy == "map_encoder_only":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.goal_encoder.parameters():         # OSM map 인코더 (9ch)
            p.requires_grad = True
        for p in model.compress_obs_enc_map.parameters(): # map 임베딩 projector
            p.requires_grad = True
        # film_model / language 관련은 이미 전체 freeze에 포함됨

    elif strategy == "partial":
        for p in model.obs_encoder.parameters():          # 관측 EfficientNet freeze
            p.requires_grad = False
        for p in model.goal_encoder_img.parameters():     # goal image 인코더 freeze
            p.requires_grad = False                        # rides_11에서 사용 안 함
        for p in model.film_model.parameters():           # FiLM freeze
            p.requires_grad = False                        # language 없음, feat_text=zeros 고정
        for p in model.compress_goal_enc_lan.parameters(): # language projector freeze
            p.requires_grad = False

    elif strategy == "none":
        for p in model.parameters():
            p.requires_grad = True

    else:
        raise ValueError(f"Unknown freeze strategy: {strategy}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[freeze] strategy={strategy} | trainable={trainable:,} / total={total:,} params")


# ══════════════════════════════════════════════════════════════════════════════
# Loss 계산
# ══════════════════════════════════════════════════════════════════════════════

def compute_loss(
    action_pred:  torch.Tensor,  # (B, 8, 4) — [x, y, cos_h, sin_h]
    gt_waypoints: torch.Tensor,  # (B, 8, 2) — [x_ego, y_ego] 정규화됨
    smooth_weight: float = 0.1,
) -> tuple:
    """
    L_wp:     smooth_l1_loss(pred_xy, gt_xy)
              smooth_l1 = L1/L2 중간, outlier에 robust

    L_smooth: temporal acceleration penalty
              v[t] = wp[t] - wp[t-1], acc = v[t] - v[t-1]
              급격한 방향 전환 억제

    L_total = L_wp + smooth_weight * L_smooth
    """
    pred_xy = action_pred[:, :, :2]  # (B, 8, 2)

    L_wp = F.smooth_l1_loss(pred_xy, gt_waypoints)

    velocity = pred_xy[:, 1:, :] - pred_xy[:, :-1, :]   # (B, 7, 2)
    accel    = velocity[:, 1:, :] - velocity[:, :-1, :]  # (B, 6, 2)
    L_smooth = accel.pow(2).mean()

    L_total = L_wp + smooth_weight * L_smooth
    return L_total, L_wp, L_smooth


# ══════════════════════════════════════════════════════════════════════════════
# 평가 지표: ADE / FDE
# ══════════════════════════════════════════════════════════════════════════════

def compute_ade_fde(
    action_pred:  torch.Tensor,  # (B, 8, 4)
    gt_waypoints: torch.Tensor,  # (B, 8, 2) 정규화됨
) -> tuple:
    """
    ADE (Average Displacement Error): 8개 waypoint 전체 평균 거리 오차 (미터)
    FDE (Final Displacement Error):   마지막 waypoint(2.4초 후) 거리 오차 (미터)

    정규화 해제: × METRIC_WAYPOINT_SPACING (0.125m) → 실제 미터
    → ADE 0.5m: 평균적으로 0.5m 오차, FDE 1.0m: 2.4초 후 1.0m 오차
    """
    pred_xy = action_pred[:, :, :2].detach()  # (B, 8, 2)
    gt_xy   = gt_waypoints.detach()           # (B, 8, 2)

    # 정규화 해제: 모델 출력과 gt 모두 /METRIC_WAYPOINT_SPACING 상태
    # 실제 미터로 환산
    pred_m = pred_xy * METRIC_WAYPOINT_SPACING  # (B, 8, 2) 미터
    gt_m   = gt_xy   * METRIC_WAYPOINT_SPACING  # (B, 8, 2) 미터

    # 각 waypoint별 L2 거리
    dist = torch.norm(pred_m - gt_m, dim=-1)  # (B, 8)

    # ADE: 8개 waypoint 평균
    ade = dist.mean().item()
    # FDE: 마지막 waypoint (fi+24, 약 2.4초 후)
    fde = dist[:, -1].mean().item()

    return ade, fde


# ══════════════════════════════════════════════════════════════════════════════
# Batch 준비
# ══════════════════════════════════════════════════════════════════════════════

def prepare_batch(batch: dict, device: torch.device) -> dict:
    """
    rides11_dataset.__getitem__ 반환값 → OmniVLA_edge.forward() 입력 변환.

    rides11_dataset 반환:
      obs_stack    (B, 18, 96, 96) — ctx5~ctx1 + obs_cur (6프레임 × 3ch)
      map_images   (B, 3,  96, 96) — OSM 맵 1장 (odom3ch 모델)
      gt_waypoints (B, 8, 2)       — ego-frame waypoints / METRIC_WAYPOINT_SPACING

    forward() 요구:
      obs_img     (B, 18, 96, 96)  ← obs_stack 그대로
      goal_pose   (B, 4)           ← zeros (modality_id=0으로 mask됨)
      map_images  (B, 3,  96, 96)  ← map_images 그대로 (3ch)
      goal_img    (B, 3,  96, 96)  ← obs_cur (forward 내부에서 cat으로 6ch)
      goal_mask   (B,)             ← 고정 0 ("satellite only")
      feat_text   (B, 512)         ← zeros (language 없음, mask됨)
      current_img (B, 3, 224, 224) ← obs_cur bilinear 224×224
    """
    B = batch["obs_stack"].shape[0]

    obs_stack  = batch["obs_stack"].to(device)    # (B, 18, 96, 96)
    map_images = batch["map_images"].to(device)   # (B, 9, 96, 96)
    gt_wp      = batch["gt_waypoints"].to(device) # (B, 8, 2)

    # 현재 프레임: obs_stack의 마지막 3채널
    obs_cur = obs_stack[:, -3:, :, :]  # (B, 3, 96, 96)

    # goal_pose: zeros (modality_id=0 → attention mask로 차단)
    goal_pose = torch.zeros(B, 4, device=device)

    # goal_img: obs_cur (goal image 없음, forward 내부에서 cat → 6ch)
    goal_img = obs_cur  # (B, 3, 96, 96)

    # goal_mask: 0 고정 ("satellite only")
    goal_mask = torch.zeros(B, dtype=torch.long, device=device)

    # feat_text: zeros (CLIP ViT-B/32 텍스트 피처 dim=512)
    feat_text = torch.zeros(B, 512, device=device)

    # current_img: FiLM 컨디셔닝용 224×224 이미지
    current_img = F.interpolate(
        obs_cur, size=(224, 224), mode="bilinear", align_corners=False
    )  # (B, 3, 224, 224)

    return {
        "obs_img":      obs_stack,
        "goal_pose":    goal_pose,
        "map_images":   map_images,
        "goal_img":     goal_img,
        "goal_mask":    goal_mask,
        "feat_text":    feat_text,
        "current_img":  current_img,
        "gt_waypoints": gt_wp,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 시각화: pred vs gt waypoint를 OSM 맵 위에 오버레이
# ══════════════════════════════════════════════════════════════════════════════

def visualize_waypoints(
    obs_stack:    torch.Tensor,  # (B, 18, 96, 96) — 현재 카메라 이미지 포함
    map_images:   torch.Tensor,  # (B, 9,  96, 96) — OSM 맵
    action_pred:  torch.Tensor,  # (B, 8, 4)
    gt_waypoints: torch.Tensor,  # (B, 8, 2) — /METRIC_WAYPOINT_SPACING 정규화됨
    ep_ids:       torch.Tensor = None,   # (B,) episode_index
    fi_ids:       torch.Tensor = None,   # (B,) frame_index
    seg_ids:      torch.Tensor = None,   # (B,) segment
    n_vis: int = 4,
) -> plt.Figure:
    """
    각 샘플마다 3개 서브플롯:
      Col 1: 현재 카메라 이미지 (obs_stack의 마지막 3ch)
      Col 2: Trajectory plot (inference/run_omnivla_edge.py의 save_robot_behavior와 동일)
              - GT:   초록선 (x=forward, y=-left)
              - Pred: 빨간선
              - Ego:  파란 점 (origin)
              - xlim=(-3,3), ylim=(-0.1,10) — inference 코드와 동일
      Col 3: OSM 맵 위에 waypoint 오버레이
              - OSM 맵: 첫 3채널 (정규화 역변환)
              - GT:   초록 점선 (OSM 맵 빨간 미래경로와 다른 스타일)
              - Pred: 빨간 실선 (OSM 맵 미래경로보다 굵게)
              - Ego:  흰 별 (이미지 중심)

    좌표계 (trajectory plot):
      waypoints는 /METRIC_WAYPOINT_SPACING 정규화 → × 0.125 = 실제 미터
      inference 코드: x_seq=wp[:,0](forward), y_seq_inv=-wp[:,1](right)
      plot: plt.plot(y_seq_inv, x_seq) → x축=좌우, y축=전방

    좌표계 (OSM 맵 오버레이):
      이미지 중심(cx,cy) = ego 위치
      MAP_RANGE_M=25m → 전체 96px=50m → PX_PER_M = 96/50 = 1.92
      pixel_x = cx - wp_m[:,1] * PX_PER_M  (y=left → 왼쪽이 +x)
      pixel_y = cy - wp_m[:,0] * PX_PER_M  (x=forward → 위쪽이 +y이므로 반전)
    """
    IMG_MEAN    = torch.tensor([0.485, 0.456, 0.406])
    IMG_STD     = torch.tensor([0.229, 0.224, 0.225])
    MAP_RANGE_M = 25.0  # osm_map_generator.py의 MAP_RANGE_M

    B = min(n_vis, obs_stack.shape[0])
    # 행=샘플, 열=3개 서브플롯
    fig, axes = plt.subplots(B, 3, figsize=(12, 4 * B))
    if B == 1:
        axes = axes[np.newaxis, :]  # (1,3) 보장

    for i in range(B):
        # ── Col 1: 현재 카메라 이미지 ─────────────────────────────────────────
        ax_img = axes[i, 0]
        # obs_stack의 마지막 3채널 = 현재 프레임
        cur_img = obs_stack[i, -3:].cpu()  # (3, 96, 96)
        cur_img = cur_img * IMG_STD[:, None, None] + IMG_MEAN[:, None, None]
        cur_img = cur_img.permute(1, 2, 0).clamp(0, 1).numpy()
        ax_img.imshow(cur_img)
        # 제목에 실제 데이터 식별자 표시
        if ep_ids is not None:
            ep  = ep_ids[i].item()
            fi  = fi_ids[i].item()
            seg = seg_ids[i].item()
            id_str = f"ep{ep:04d}_seg{seg:02d}_fi{fi:06d}"
        else:
            id_str = f"sample_{i}"
        ax_img.set_title(f"{id_str}\nCurrent Image", fontsize=8)
        ax_img.axis("off")

        # ── 공통: waypoint 미터 환산 ──────────────────────────────────────────
        # gt/pred 모두 /METRIC_WAYPOINT_SPACING 정규화됨 → × 0.125 = 미터
        gt   = gt_waypoints[i].cpu().numpy()             # (8,2)
        pred = action_pred[i, :, :2].detach().cpu().numpy()  # (8,2)
        gt_m   = gt   * METRIC_WAYPOINT_SPACING  # 실제 미터
        pred_m = pred * METRIC_WAYPOINT_SPACING

        # ── Col 2: Trajectory plot (inference 스타일) ──────────────────────────
        ax_traj = axes[i, 1]

        # GT: x=forward(y축), y=-left=right(x축), origin(0,0) 삽입
        gt_x_plot   = -gt_m[:, 1]              # right = -left
        gt_y_plot   =  gt_m[:, 0]              # forward
        ax_traj.plot(
            np.insert(gt_x_plot,   0, 0.0),
            np.insert(gt_y_plot,   0, 0.0),
            linewidth=2.5, markersize=8, marker='o', color='green', label='GT'
        )

        # Pred
        pred_x_plot = -pred_m[:, 1]
        pred_y_plot =  pred_m[:, 0]
        ax_traj.plot(
            np.insert(pred_x_plot, 0, 0.0),
            np.insert(pred_y_plot, 0, 0.0),
            linewidth=2.5, markersize=8, marker='o', color='red',   label='Pred'
        )

        # Ego origin
        ax_traj.plot(0, 0, marker='o', color='blue', markersize=12, zorder=5)

        # inference 코드와 동일한 축 범위
        ax_traj.set_xlim(-3.0, 3.0)
        ax_traj.set_ylim(-0.1, 10.0)
        ax_traj.set_xlabel("Left/Right (m)", fontsize=8)
        ax_traj.set_ylabel("Forward (m)",    fontsize=8)
        ax_traj.set_title(f"{id_str}\nTrajectory (satellite only)", fontsize=8)
        ax_traj.legend(fontsize=8)
        ax_traj.grid(True, alpha=0.3)
        ax_traj.tick_params(labelsize=7)

        # ── Col 3: OSM 맵 오버레이 ────────────────────────────────────────────
        ax_osm = axes[i, 2]

        # OSM 맵 복원 (첫 3채널)
        osm = map_images[i, :3].cpu()
        osm = osm * IMG_STD[:, None, None] + IMG_MEAN[:, None, None]
        osm = osm.permute(1, 2, 0).clamp(0, 1).numpy()
        ax_osm.imshow(osm)

        H, W       = osm.shape[:2]
        cx, cy     = W / 2.0, H / 2.0
        PX_PER_M   = H / (MAP_RANGE_M * 2.0)  # 96/(25×2) = 1.92 px/m

        # GT — 초록 점선 (OSM 맵 빨간 경로와 스타일로 구분)
        gt_px = cx - gt_m[:, 1] * PX_PER_M   # y(left) → pixel x 반전
        gt_py = cy - gt_m[:, 0] * PX_PER_M   # x(forward) → pixel y 반전(위=전방)
        ax_osm.plot(gt_px, gt_py, '--o', color='lime',
                    markersize=4, linewidth=2, label='GT')

        # Pred — 흰 실선 (OSM 맵 위에서 잘 보이도록)
        pred_px = cx - pred_m[:, 1] * PX_PER_M
        pred_py = cy - pred_m[:, 0] * PX_PER_M
        ax_osm.plot(pred_px, pred_py, '-o', color='white',
                    markersize=4, linewidth=2, label='Pred',
                    markeredgecolor='red', markeredgewidth=1)

        # Ego 마커 (파란 별, 검은 테두리)
        ax_osm.plot(cx, cy, '*', color='cyan', markersize=12,
                    markeredgecolor='black', markeredgewidth=0.8)

        ax_osm.set_title(f"{id_str}\nOSM Map Overlay", fontsize=8)
        ax_osm.legend(fontsize=7, loc='lower right')
        ax_osm.axis("off")

    fig.suptitle("OmniVLA-Edge Fine-tuning Validation", fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 학습 한 epoch
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model:        nn.Module,
    loader:       DataLoader,
    optimizer:    torch.optim.Optimizer,
    device:       torch.device,
    epoch:        int,
    model_params: dict,
    smooth_weight: float,
    log_freq:     int,
    use_wandb:    bool,
    global_step:  int,
) -> tuple:
    """한 epoch 학습. (train_loss, train_wp_loss, global_step) 반환."""
    model.train()
    total_loss_sum = 0.0
    wp_loss_sum    = 0.0

    for i, batch in enumerate(loader):
        inp = prepare_batch(batch, device)

        # forward — action_pred: (B,8,4), dist_pred: (B,1) 거리예측(미사용)
        action_pred, _, _ = model(
            inp["obs_img"],
            inp["goal_pose"],
            inp["map_images"],
            inp["goal_img"],
            inp["goal_mask"],
            inp["feat_text"],
            inp["current_img"],
        )

        L_total, L_wp, L_smooth = compute_loss(
            action_pred, inp["gt_waypoints"], smooth_weight
        )

        optimizer.zero_grad()
        L_total.backward()
        # gradient clipping — 학습 초기 불안정 방지 (max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss_sum += L_total.item()
        wp_loss_sum    += L_wp.item()
        global_step    += 1

        # iteration 단위 wandb 로그 (loss 곡선)
        if use_wandb:
            wandb.log({
                "train/loss_total":  L_total.item(),
                "train/loss_wp":     L_wp.item(),
                "train/loss_smooth": L_smooth.item(),
            }, step=global_step)

        if (i + 1) % log_freq == 0:
            avg_total = total_loss_sum / (i + 1)
            avg_wp    = wp_loss_sum    / (i + 1)
            print(f"  [epoch {epoch} | iter {i+1}/{len(loader)}] "
                  f"loss={avg_total:.4f}  wp={avg_wp:.4f}  smooth={L_smooth.item():.4f}")

    return total_loss_sum / len(loader), wp_loss_sum / len(loader), global_step


# ══════════════════════════════════════════════════════════════════════════════
# Validation: loss + ADE/FDE + 시각화
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(
    model:        nn.Module,
    loader:       DataLoader,
    device:       torch.device,
    model_params: dict,
    smooth_weight: float,
    epoch:        int,
    use_wandb:    bool,
    vis_freq:     int,
    save_dir:     Path,
) -> tuple:
    """
    Validation loss, ADE, FDE 계산 및 시각화.
    (val_loss, ade, fde) 반환.
    """
    model.eval()
    total_loss_sum = 0.0
    ade_sum        = 0.0
    fde_sum        = 0.0
    n_batches      = 0

    vis_batch = None  # 시각화용 첫 번째 배치 저장

    for batch in loader:
        inp = prepare_batch(batch, device)
        action_pred, _, _ = model(
            inp["obs_img"], inp["goal_pose"], inp["map_images"],
            inp["goal_img"], inp["goal_mask"], inp["feat_text"], inp["current_img"],
        )

        L_total, _, _ = compute_loss(action_pred, inp["gt_waypoints"], smooth_weight)
        ade, fde      = compute_ade_fde(action_pred, inp["gt_waypoints"])

        total_loss_sum += L_total.item()
        ade_sum        += ade
        fde_sum        += fde
        n_batches      += 1

        # 시각화용 배치 저장 (첫 번째 배치만)
        if vis_batch is None:
            vis_batch = (
                inp["obs_img"], inp["map_images"], action_pred, inp["gt_waypoints"],
                batch["episode_index"], batch["frame_index"], batch["segment"],
            )

    val_loss = total_loss_sum / n_batches
    ade      = ade_sum        / n_batches
    fde      = fde_sum        / n_batches

    # epoch 단위 wandb 로그 (val 지표)
    if use_wandb:
        log_dict = {
            "val/loss":  val_loss,
            "val/ADE_m": ade,   # 미터 단위 ADE (낮을수록 좋음)
            "val/FDE_m": fde,   # 미터 단위 FDE (낮을수록 좋음)
            "epoch":     epoch,
        }

        # vis_freq epoch마다 waypoint 오버레이 이미지를 wandb에 업로드
        if epoch % vis_freq == 0 and vis_batch is not None:
            obs_imgs, map_imgs, pred, gt, ep_ids, fi_ids, seg_ids = vis_batch
            fig = visualize_waypoints(obs_imgs, map_imgs, pred, gt,
                                      ep_ids=ep_ids, fi_ids=fi_ids, seg_ids=seg_ids,
                                      n_vis=4)

            # wandb.Image로 변환해서 업로드
            log_dict["val/waypoint_vis"] = wandb.Image(fig)

            # 로컬에도 저장 (save_dir/vis/epoch_XXX.png)
            vis_dir = save_dir / "vis"
            vis_dir.mkdir(exist_ok=True)
            fig.savefig(vis_dir / f"epoch_{epoch:03d}.png", dpi=100)
            plt.close(fig)

        wandb.log(log_dict)

    return val_loss, ade, fde


# ══════════════════════════════════════════════════════════════════════════════
# Test set 평가: trajectory 시각화 전체 저장
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_test(
    model:        nn.Module,
    loader:       DataLoader,
    device:       torch.device,
    smooth_weight: float,
    save_dir:     Path,
) -> None:
    """
    Test set 전체 배치에 대해 ADE/FDE 계산 + trajectory 시각화 저장.
    save_dir/test_vis/ 아래에 batch_XXXX.png로 저장.
    """
    model.eval()
    vis_dir = save_dir / "test_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    total_loss_sum = 0.0
    ade_sum = 0.0
    fde_sum = 0.0
    n_batches = 0

    for i, batch in enumerate(loader):
        inp = prepare_batch(batch, device)
        action_pred, _, _ = model(
            inp["obs_img"], inp["goal_pose"], inp["map_images"],
            inp["goal_img"], inp["goal_mask"], inp["feat_text"], inp["current_img"],
        )

        L_total, _, _ = compute_loss(action_pred, inp["gt_waypoints"], smooth_weight)
        ade, fde      = compute_ade_fde(action_pred, inp["gt_waypoints"])

        total_loss_sum += L_total.item()
        ade_sum        += ade
        fde_sum        += fde
        n_batches      += 1

        # 모든 배치 시각화 저장
        fig = visualize_waypoints(
            inp["obs_img"], inp["map_images"], action_pred, inp["gt_waypoints"],
            ep_ids=batch["episode_index"], fi_ids=batch["frame_index"],
            seg_ids=batch["segment"], n_vis=4,
        )
        fig.savefig(vis_dir / f"batch_{i:04d}.png", dpi=80)
        plt.close(fig)

        if (i + 1) % 20 == 0:
            print(f"  [test] batch {i+1}/{len(loader)}  "
                  f"ADE={ade:.3f}m  FDE={fde:.3f}m")

    avg_loss = total_loss_sum / n_batches
    avg_ade  = ade_sum        / n_batches
    avg_fde  = fde_sum        / n_batches
    print(f"\n[test] loss={avg_loss:.4f}  ADE={avg_ade:.3f}m  FDE={avg_fde:.3f}m")
    print(f"[test] 시각화 저장 완료: {vis_dir}  ({n_batches}장)")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(cfg: dict):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"[main] device: {device}")

    # wandb 초기화
    use_wandb = cfg.get("use_wandb", TRAIN_CONFIG["use_wandb"])
    if use_wandb:
        wandb.init(
            project=cfg.get("wandb_project", TRAIN_CONFIG["wandb_project"]),
            config=cfg,   # 모든 하이퍼파라미터를 wandb config에 저장
            name=cfg.get("run_name", None),  # run 이름 (없으면 wandb 자동 생성)
        )

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    model, text_encoder = load_model(
        ckpt_path=cfg.get("ckpt_path", str(CKPT_PATH)),
        model_params=MODEL_PARAMS,
        device=device,
    )
    model        = model.to(device)
    text_encoder = text_encoder.to(device).eval()  # text_encoder는 학습하지 않음

    # ── eval_only: 체크포인트 로드 후 test set 평가만 실행 ────────────────────
    if cfg.get("eval_only", False):
        eval_ckpt = cfg.get("eval_ckpt", str(save_dir / "best.pth"))
        print(f"[eval_only] Loading: {eval_ckpt}")
        ckpt = torch.load(eval_ckpt, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        # dataset/loader는 아래에서 구성되므로 여기서는 플래그만 확인
        # (실제 evaluate_test 호출은 loader 구성 후)

    # ── Freeze 전략 ───────────────────────────────────────────────────────────
    if not cfg.get("eval_only", False):
        apply_freeze(model, cfg.get("freeze", TRAIN_CONFIG["freeze"]))

    # ── 데이터셋 ──────────────────────────────────────────────────────────────
    dataset = Rides11Dataset(
        arrow_path=cfg.get("arrow_path",  str(ARROW_PATH)),
        scores_path=cfg.get("scores_path", str(SCORES_PATH)),
        osm_root=cfg.get("osm_root",       str(OSM_ROOT)),
        video_root=cfg.get("video_root",   str(VIDEO_ROOT)),
    )
    print(f"[main] Dataset size: {len(dataset):,}")

    val_ratio  = cfg.get("val_ratio",  TRAIN_CONFIG["val_ratio"])
    test_ratio = cfg.get("test_ratio", TRAIN_CONFIG["test_ratio"])
    val_size   = int(len(dataset) * val_ratio)
    test_size  = int(len(dataset) * test_ratio)
    train_size = len(dataset) - val_size - test_size
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"[main] train={train_size:,}  val={val_size:,}  test={test_size:,}")

    bs = cfg.get("batch_size", TRAIN_CONFIG["batch_size"])
    nw = cfg.get("num_workers", TRAIN_CONFIG["num_workers"])

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, drop_last=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, drop_last=False, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                              num_workers=nw, drop_last=False, persistent_workers=True)

    # ── eval_only: test set 평가 후 종료 ─────────────────────────────────────
    if cfg.get("eval_only", False):
        smooth_w = cfg.get("smooth_weight", TRAIN_CONFIG["smooth_weight"])
        evaluate_test(model, test_loader, device, smooth_w, save_dir)
        if use_wandb:
            wandb.finish()
        return

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=cfg.get("lr", TRAIN_CONFIG["lr"]),
        weight_decay=cfg.get("weight_decay", TRAIN_CONFIG["weight_decay"]),
    )
    # CosineAnnealingLR: lr을 epoch에 따라 cosine으로 감소 (T_max=epochs)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.get("epochs", TRAIN_CONFIG["epochs"]),
    )

    # ── 저장 디렉토리 ─────────────────────────────────────────────────────────
    save_dir = Path(cfg.get("save_dir", str(SAVE_DIR)))
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs     = cfg.get("epochs",        TRAIN_CONFIG["epochs"])
    save_freq  = cfg.get("save_freq",     TRAIN_CONFIG["save_freq"])
    log_freq   = cfg.get("log_freq",      TRAIN_CONFIG["log_freq"])
    vis_freq   = cfg.get("vis_freq",      TRAIN_CONFIG["vis_freq"])
    smooth_w   = cfg.get("smooth_weight", TRAIN_CONFIG["smooth_weight"])

    best_val_loss = float("inf")
    global_step   = 0
    start_epoch   = 1

    # ── Resume 처리 ───────────────────────────────────────────────────────────
    resume_ckpt = cfg.get("resume_ckpt", None)
    if resume_ckpt is not None and Path(resume_ckpt).exists():
        print(f"[resume] Loading checkpoint: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            # epoch별 체크포인트 (optimizer/scheduler 포함)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch   = ckpt["epoch"] + 1
            global_step   = ckpt.get("global_step", 0)
            best_val_loss = ckpt.get("val_loss", float("inf"))
            print(f"[resume] Resumed from epoch {ckpt['epoch']}  "
                  f"val_loss={best_val_loss:.4f}  start_epoch={start_epoch}")
        else:
            # best.pth (model state_dict만) — optimizer/scheduler는 reset
            model.load_state_dict(ckpt)
            # --start_epoch으로 지정한 epoch부터 시작하도록 scheduler catch-up
            override_epoch = cfg.get("start_epoch", 1)
            start_epoch = override_epoch
            for _ in range(override_epoch - 1):
                scheduler.step()
            best_val_loss = cfg.get("resume_val_loss", float("inf"))
            print(f"[resume] Loaded model weights from best.pth  "
                  f"start_epoch={start_epoch}  best_val_loss={best_val_loss:.4f}")

    # ── 학습 루프 ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n[Epoch {epoch}/{epochs}]")

        train_loss, train_wp, global_step = train_one_epoch(
            model, train_loader, optimizer, device,
            epoch, MODEL_PARAMS, smooth_w, log_freq, use_wandb, global_step,
        )

        val_loss, ade, fde = validate(
            model, val_loader, device, MODEL_PARAMS, smooth_w,
            epoch, use_wandb, vis_freq, save_dir,
        )

        # lr scheduler step (epoch 단위)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # epoch 요약 출력
        print(f"  → train_loss={train_loss:.4f}  train_wp={train_wp:.4f}  "
              f"val_loss={val_loss:.4f}  ADE={ade:.3f}m  FDE={fde:.3f}m  "
              f"lr={current_lr:.2e}")

        # wandb: epoch 단위 lr 로그
        if use_wandb:
            wandb.log({"train/lr": current_lr, "epoch": epoch})

        # best 체크포인트 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / "best.pth")
            print(f"  ★ best model saved  val_loss={best_val_loss:.4f}  "
                  f"ADE={ade:.3f}m  FDE={fde:.3f}m")

        # 주기적 체크포인트 저장 (optimizer/scheduler 포함, resume 가능)
        if epoch % save_freq == 0:
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss":             val_loss,
                "ade":                  ade,
                "fde":                  fde,
                "global_step":          global_step,
            }, save_dir / f"epoch_{epoch:03d}.pth")

    print(f"\n[Done] Best val_loss: {best_val_loss:.4f}")
    print(f"       Checkpoints: {save_dir}")

    if use_wandb:
        wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="YAML config 경로")
    parser.add_argument("--resume_ckpt", type=str, default=None,
                        help="Resume할 체크포인트 경로 (best.pth 또는 epoch_XXX.pth)")
    parser.add_argument("--start_epoch", type=int, default=None,
                        help="best.pth resume 시 시작 epoch 번호 (예: 4)")
    parser.add_argument("--resume_val_loss", type=float, default=None,
                        help="best.pth resume 시 현재까지의 best val_loss (예: 0.2624)")
    parser.add_argument("--eval_only", action="store_true",
                        help="학습 없이 test set 평가 + trajectory 시각화만 실행")
    parser.add_argument("--eval_ckpt", type=str, default=None,
                        help="eval_only 시 로드할 체크포인트 경로 (기본: save_dir/best.pth)")
    args = parser.parse_args()

    cfg = dict(TRAIN_CONFIG)
    if args.config is not None:
        with open(args.config) as f:
            user_cfg = yaml.safe_load(f)
        cfg.update(user_cfg)
    if args.resume_ckpt is not None:
        cfg["resume_ckpt"] = args.resume_ckpt
    if args.start_epoch is not None:
        cfg["start_epoch"] = args.start_epoch
    if args.resume_val_loss is not None:
        cfg["resume_val_loss"] = args.resume_val_loss
    if args.eval_only:
        cfg["eval_only"] = True
    if args.eval_ckpt is not None:
        cfg["eval_ckpt"] = args.eval_ckpt

    main(cfg)
