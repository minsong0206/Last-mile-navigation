"""
MBRA Reannotation Script
gnm_format/ → processed_gnm/

각 ride에 대해:
1. front camera jpg 복사
2. MBRA 모델로 (obs, goal) → trajectory inference
3. 누적 pos/yaw로 odom map (BEV top-down) 생성 → odom_{i}.jpg
4. traj_data.pkl 저장 (position/yaw = MBRA 출력)
"""

import os
import sys
import shutil
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/train")

from vint_train.models.exaug.exaug import ExAug_dist_delay

# ── 상수 ──────────────────────────────────────────────────────────────────────
MBRA_CKPT = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/train/logs/frodobot-gnm/frodobot-gnm_2026_04_27_10_31_23/mbra.pth"
GNM_ROOT  = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/FrodoBots-2K/gnm_format"
OUT_ROOT  = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/FrodoBots-2K/processed_gnm"

IMAGE_SIZE   = (96, 96)   # W, H (MBRA config 기준)
CONTEXT_SIZE = 5          # MBRA config
GOAL_HORIZON = 10         # obs → goal 프레임 간격 (gnm_format ~0.45m/frame → ~4.5m 앞)
DT           = 0.333      # twist_to_pose_diff에서 사용하는 dt
METRIC_SPACING = 0.25 * 0.5  # NoMaD action normalization factor

# odom map 파라미터
MAP_SIZE_M  = 20.0   # 지도 범위 (미터)
MAP_PX      = 256    # 지도 이미지 크기 (픽셀)
ROBOT_TRAIL = True   # 지나온 경로 그리기

TRANSFORM = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── 모델 로드 ──────────────────────────────────────────────────────────────────
def load_mbra(ckpt_path, device):
    model = ExAug_dist_delay(
        context_size=CONTEXT_SIZE,
        len_traj_pred=8,
        learn_angle=True,
        obs_encoder="efficientnet-b0",
        obs_encoding_size=1024,
        late_fusion=False,
        mha_num_attention_heads=4,
        mha_num_attention_layers=4,
        mha_ff_dim_factor=4,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # state_dict 키 prefix 처리
    if any(k.startswith("module.") for k in ckpt.keys()):
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model.eval().to(device)
    return model


# ── 이미지 로드 헬퍼 ───────────────────────────────────────────────────────────
def load_img_tensor(path, device):
    img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return TRANSFORM(t).to(device)


# ── 속도 적분 (train_utils.robot_pos_model_fix 동일 로직) ───────────────────────
def sinc_apx(angle):
    return torch.sin(torch.tensor(torch.pi) * angle + 1e-9) / (torch.tensor(torch.pi) * angle + 1e-9)

def twist_to_pose(v, w, dt=DT):
    theta = -w * dt
    z = v * dt * sinc_apx(-theta / torch.pi)
    x = -v * dt * sinc_apx(-theta / (2 * torch.pi)) * torch.sin(-theta / 2)
    return x, z, theta

def integrate_velocities(linear_vel, angular_vel):
    """
    linear_vel, angular_vel: (horizon,) tensors
    반환: positions (horizon, 2), yaws (horizon,) — 카메라 좌표계
    """
    horizon = linear_vel.shape[0]
    Tacc = torch.eye(4)
    px_list, pz_list, pyaw_list = [], [], []

    for i in range(horizon):
        x, z, yaw = twist_to_pose(linear_vel[i], angular_vel[i])
        Todom = torch.zeros(4, 4)
        Todom[0, 0] = torch.cos(yaw)
        Todom[0, 2] = torch.sin(yaw)
        Todom[1, 1] = 1.0
        Todom[2, 0] = -torch.sin(yaw)
        Todom[2, 2] = torch.cos(yaw)
        Todom[0, 3] = x
        Todom[2, 3] = z
        Todom[3, 3] = 1.0
        Tacc = Tacc @ Todom

        pyaw_list.append(torch.arctan(Tacc[0, 2] / (Tacc[0, 0] + 1e-9)).item())
        px_list.append(Tacc[0, 3].item())
        pz_list.append(Tacc[2, 3].item())

    # 카메라 좌표 (x=옆, z=앞) → 로봇 좌표 (x=앞, y=왼쪽)
    positions = np.array([[pz, -px] for px, pz in zip(px_list, pz_list)])
    yaws = np.array(pyaw_list)
    return positions, yaws


# ── odom map 생성 ─────────────────────────────────────────────────────────────
def make_odom_map(all_positions, all_yaws, curr_idx, map_size_m=MAP_SIZE_M, map_px=MAP_PX):
    """
    heading-up BEV odom map
    - 흰 배경
    - 지나온 경로: 회색 선 (두께 6)
    - 앞으로 갈 경로: 파란 굵은 선 (두께 8)
    - 현재 위치: 파란 원, 항상 화면 중앙 고정, 진행방향=위

    좌표계:
      position[:,0] = 로봇 전방 (forward)
      position[:,1] = 로봇 왼쪽 (left)
      yaw: 반시계 양수, 0 = 전방

    heading-up 변환:
      world를 -curr_yaw 회전 → 로봇 전방이 화면 위(+y_screen 감소 방향)
      rx (회전후 전방) → px_y 감소 (위)
      ry (회전후 왼쪽) → px_x 감소 (왼)
    """
    img = np.ones((map_px, map_px, 3), dtype=np.uint8) * 255

    curr_pos = all_positions[curr_idx]
    curr_yaw = all_yaws[curr_idx]
    scale = map_px / map_size_m

    def to_px(pos):
        rel = pos - curr_pos
        c, s = np.cos(-curr_yaw), np.sin(-curr_yaw)
        rx =  c * rel[0] - s * rel[1]   # 회전 후 전방 성분
        ry =  s * rel[0] + c * rel[1]   # 회전 후 왼쪽 성분
        # 전방=화면 위(-y), 왼쪽=화면 왼(-x)
        px_x = int(map_px / 2 - ry * scale)
        px_y = int(map_px / 2 - rx * scale)
        return (px_x, px_y)

    # 지나온 경로 (회색, 두께 6)
    for i in range(1, curr_idx + 1):
        p1 = to_px(all_positions[i - 1])
        p2 = to_px(all_positions[i])
        cv2.line(img, p1, p2, (180, 180, 180), 6)

    # 앞으로 갈 경로 (파란, 두께 8)
    future_pts = [to_px(all_positions[i]) for i in range(curr_idx, len(all_positions))]
    if len(future_pts) >= 2:
        for i in range(len(future_pts) - 1):
            cv2.line(img, future_pts[i], future_pts[i + 1], (200, 80, 0), 8)  # BGR blue

    # 현재 위치: 원 (항상 화면 중앙)
    cp = (map_px // 2, map_px // 2)
    cv2.circle(img, cp, 10, (200, 80, 0), -1)
    cv2.circle(img, cp, 10, (255, 255, 255), 2)

    return img


# ── 단일 ride 처리 ─────────────────────────────────────────────────────────────
@torch.no_grad()
def process_ride(ride_name, model, device, out_root, gnm_root, goal_horizon):
    in_dir  = os.path.join(gnm_root, ride_name)
    out_dir = os.path.join(out_root, ride_name)
    os.makedirs(out_dir, exist_ok=True)

    # jpg 파일 목록
    jpgs = sorted([f for f in os.listdir(in_dir) if f.endswith(".jpg")],
                  key=lambda f: int(f.replace(".jpg", "")))
    N = len(jpgs)
    if N < CONTEXT_SIZE + 2:
        print(f"  skip {ride_name}: too few frames ({N})")
        return False

    # jpg 복사
    for jpg in jpgs:
        shutil.copy(os.path.join(in_dir, jpg), os.path.join(out_dir, jpg))

    # 이미지 텐서 사전 로드
    imgs = [load_img_tensor(os.path.join(in_dir, jpg), device) for jpg in jpgs]

    # MBRA inference: 각 프레임 i에 대해 goal = min(i+goal_horizon, N-1)
    rsize = torch.tensor([[[0.3]]]).to(device)
    delay = torch.zeros(1, 1, 1).to(device)

    all_positions = np.zeros((N, 2))  # 누적 절대 좌표
    all_yaws      = np.zeros(N)
    all_actions   = np.zeros((N, 8, 4))  # (N, horizon, [x, y, cos, sin])

    # 첫 프레임은 원점
    # 각 프레임의 절대 pos = 이전 프레임에서 MBRA가 예측한 1-step 이동량 누적
    # (단순화: goal_horizon 스텝 중 첫 번째 스텝만 사용해 누적)
    cum_pos = np.array([0.0, 0.0])
    cum_yaw = 0.0

    for i in range(N):
        all_positions[i] = cum_pos
        all_yaws[i]      = cum_yaw

        goal_idx = min(i + goal_horizon, N - 1)

        # obs: context_size+1 장 (과거 ~ 현재)
        obs_start = max(0, i - CONTEXT_SIZE)
        obs_frames = []
        for j in range(obs_start, i + 1):
            obs_frames.append(imgs[j])
        # context_size 부족하면 첫 프레임 반복 패딩
        while len(obs_frames) < CONTEXT_SIZE + 1:
            obs_frames.insert(0, obs_frames[0])
        obs_tensor = torch.cat(obs_frames, dim=0).unsqueeze(0)  # (1, 3*(ctx+1), H, W)

        goal_tensor = imgs[goal_idx].unsqueeze(0)  # (1, 3, H, W)

        linear_vel_old = 0.5 * torch.ones(1, 6).to(device)
        angular_vel_old = 0.0 * torch.ones(1, 6).to(device)
        vel_past = torch.cat((linear_vel_old, angular_vel_old), dim=1).unsqueeze(2)

        linear_vel, angular_vel, _ = model(obs_tensor, goal_tensor, rsize, delay, vel_past)
        # linear_vel, angular_vel: (1, 8)

        positions, yaws = integrate_velocities(
            linear_vel[0].cpu(), angular_vel[0].cpu()
        )
        # positions: (8, 2) 로봇 좌표 기준 상대 위치

        # action 저장 (NoMaD 형식: [x/spacing, y/spacing, cos(yaw), sin(yaw)])
        actions = np.zeros((8, 4))
        actions[:, 0] = positions[:, 0] / METRIC_SPACING
        actions[:, 1] = positions[:, 1] / METRIC_SPACING
        actions[:, 2] = np.cos(yaws)
        actions[:, 3] = np.sin(yaws)
        all_actions[i] = actions

        # 누적 위치 업데이트: 1-step 이동량으로 절대 좌표 갱신
        if i < N - 1:
            dx = positions[0, 0]
            dy = positions[0, 1]
            # 현재 yaw 기준으로 글로벌 좌표 변환
            cum_pos = cum_pos + np.array([
                dx * np.cos(cum_yaw) - dy * np.sin(cum_yaw),
                dx * np.sin(cum_yaw) + dy * np.cos(cum_yaw),
            ])
            cum_yaw = cum_yaw + yaws[0]

    # odom map 생성
    for i in range(N):
        omap = make_odom_map(all_positions, all_yaws, i)
        omap_rgb = cv2.cvtColor(omap, cv2.COLOR_BGR2RGB)
        Image.fromarray(omap_rgb).save(os.path.join(out_dir, f"odom_{i}.jpg"))

    # traj_data.pkl 저장
    traj_dict = {
        "position": all_positions,   # (N, 2) — NoMaD ViNT_Dataset이 기대하는 키
        "yaw":      all_yaws,         # (N,)
        "actions":  all_actions,      # (N, 8, 4)
        # 원본 GPS 데이터도 보존
    }
    # 원본 traj_data.pkl에서 GPS 정보 보존
    orig_pkl = os.path.join(in_dir, "traj_data.pkl")
    if os.path.exists(orig_pkl):
        with open(orig_pkl, "rb") as f:
            orig = pickle.load(f)
        traj_dict["pos_gps"]   = orig.get("pos", None)
        traj_dict["yaw_gps"]   = orig.get("yaw", None)
        traj_dict["linear"]    = orig.get("linear", None)
        traj_dict["angular"]   = orig.get("angular", None)
        traj_dict["timestamps"] = orig.get("timestamps", None)

    with open(os.path.join(out_dir, "traj_data.pkl"), "wb") as f:
        pickle.dump(traj_dict, f)

    return True


# ── 메인 ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnm_root",     default=GNM_ROOT)
    parser.add_argument("--out_root",     default=OUT_ROOT)
    parser.add_argument("--ckpt",         default=MBRA_CKPT)
    parser.add_argument("--goal_horizon", type=int, default=GOAL_HORIZON)
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--rides",        nargs="*", default=None,
                        help="처리할 ride 이름 목록 (미지정시 전체)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("MBRA 모델 로드 중...")
    model = load_mbra(args.ckpt, device)
    print("완료")

    os.makedirs(args.out_root, exist_ok=True)

    rides = args.rides or sorted(os.listdir(args.gnm_root))
    rides = [r for r in rides if os.path.isdir(os.path.join(args.gnm_root, r))]

    print(f"총 {len(rides)}개 ride 처리 시작")
    success = 0
    for ride in tqdm(rides):
        ok = process_ride(
            ride, model, device,
            args.out_root, args.gnm_root, args.goal_horizon
        )
        if ok:
            success += 1

    print(f"\n완료: {success}/{len(rides)} rides → {args.out_root}")


if __name__ == "__main__":
    main()
