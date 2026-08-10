"""
omnivla_edge_deploy.py

FrodoBot Mini 실로봇 배포용 OmniVLA-Edge-Odom 추론 루프.

deployment/LogoNav_frodobot.py의 FrodoBot SDK 연동 패턴(REST API: /v2/front 카메라,
/data GPS, /control 제어명령)을 그대로 재사용하되, 모델을 OmniVLA-Edge-Odom(rides_11
파인튜닝 체크포인트)으로 교체하고, 맵 입력은 build_live_map.py의 LiveMapBuilder로
실시간 생성한다 (학습 때처럼 GT 미래 GPS가 없으므로 OSRM 실시간 라우팅으로 대체).

⚠ 중요 — 아직 실로봇으로 end-to-end 테스트 안 됨. 반드시:
  1. 시뮬레이션/정지 상태에서 predicted waypoint 출력이 합리적인지 먼저 확인
  2. 저속(MAX_V를 작게)으로 개활지에서 첫 테스트
  3. e-stop 또는 SDK 긴급정지를 항상 준비해둔 상태로 진행

학습-추론 일치 확인 필수 항목 (finetune_omnivla_edge.py::prepare_batch와 반드시 동일):
  - MAP_RANGE_M: 체크포인트를 학습시킨 맵 반경과 정확히 같은 값을 --map_range로 넘길 것.
    체크포인트별 정답 값:
      checkpoints/omnivla_edge_rides11_odom/best.pth       → 25 (기본, baseline)
      checkpoints/omnivla_edge_rides11_odom_12m/best.pth   → 12
      checkpoints/omnivla_edge_rides11_odom_20m/best.pth   → 20 (교수님 피드백 반영, 얇은 경로선 ROUTE_WIDTH=2)
  - N_CTX=5, CTX_STRIDE=3프레임(~0.3초 간격), 카메라 6장(과거5+현재1)
  - modality_id / goal_mask = 0 ("map only")
  - METRIC_WAYPOINT_SPACING = 0.125 (출력 waypoint를 미터로 변환할 때)

실행 예 (20m 체크포인트 기준):
  /home/ms/uv-envs/mbra/venv/bin/python deployment/omnivla_edge_deploy.py \
      --ckpt checkpoints/omnivla_edge_rides11_odom_20m/best.pth \
      --map_range 20 \
      --goal_lat 37.5010 --goal_lon 127.0010
"""

import sys
import time
import base64
import io
import argparse
import math
from pathlib import Path
from collections import deque

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "third_party" / "omnivla" / "inference"))
sys.path.insert(0, str(REPO_ROOT / "deployment"))
from model_omnivla_edge_odom import OmniVLA_edge_odom
from build_live_map import LiveMapBuilder
from debug_web import DeploymentState, start_debug_server

FRODOBOT_BASE = "http://127.0.0.1:8000"

# rides11_dataset.py와 동일한 학습 시 상수
N_CTX = 5
CTX_STRIDE_SEC = 0.3          # 프레임 간격 (WAYPOINT_STRIDE=3 frames @ ~10Hz)
METRIC_WAYPOINT_SPACING = 0.125
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

MODEL_PARAMS = dict(
    context_size=5, len_traj_pred=8, learn_angle=True,
    obs_encoder="efficientnet-b0", obs_encoding_size=1024,
    late_fusion=False, mha_num_attention_heads=4,
    mha_num_attention_layers=4, mha_ff_dim_factor=4,
)

# ── 제어 안전 한계 (LogoNav_frodobot.py와 동일한 안전장치 재사용) ──
MAX_V = 0.3   # m/s
MAX_W = 0.3   # rad/s
DT    = 1.0 / 3.0  # 제어 루프 주기 (3Hz, 학습 waypoint 간격 0.3초와 근접)


def decode_frame(b64_str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def clip_control(linear_vel, angular_vel, maxv=MAX_V, maxw=MAX_W):
    """LogoNav_frodobot.py::policy_calc()의 클리핑 로직 재사용 —
    linear/angular 비율(rd)을 유지한 채 한계 안으로 스케일링."""
    if abs(linear_vel) <= maxv and abs(angular_vel) <= maxw:
        return linear_vel, angular_vel
    if abs(angular_vel) <= 1e-6:
        return maxv * np.sign(linear_vel), 0.0
    rd = linear_vel / angular_vel
    if abs(rd) >= maxv / maxw:
        return maxv * np.sign(linear_vel), maxv * np.sign(angular_vel) / abs(rd)
    return maxw * np.sign(linear_vel) * abs(rd), maxw * np.sign(angular_vel)


class OmniVLAEdgeDeployment:
    def __init__(self, ckpt_path, map_range_m, goal_lat, goal_lon, device=None,
                 debug_port=8080):
        self.state = DeploymentState()
        if debug_port:
            start_debug_server(self.state, port=debug_port)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = OmniVLA_edge_odom(**MODEL_PARAMS)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
        self.model.to(self.device).eval()
        print(f"[deploy] Loaded checkpoint: {ckpt_path}  (device={self.device})")

        self.map_builder = LiveMapBuilder(map_range_m=map_range_m)
        self.goal_lat, self.goal_lon = goal_lat, goal_lon
        # 경로는 배포 시작 시 1번만 계산해서 캐싱 (osmnav 구조 참고 — 매 프레임 재쿼리 안 함).
        # 시작 위치를 아직 모르므로, 첫 poll_frodobot() 이후 run()에서 set_goal() 호출.
        self._route_initialized = False

        self.obs_transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ])

        # 최근 카메라 프레임 (0.3초 간격으로 채워짐, N_CTX+1개 유지)
        self.frame_buffer = deque(maxlen=N_CTX + 1)
        self.last_frame_time = 0.0

        # 로봇이 실제로 지나온 GPS 기록 (odom map의 회색 past 선용, 최근 것만 유지)
        self.past_track = deque(maxlen=200)

    def poll_frodobot(self):
        cam = requests.get(f"{FRODOBOT_BASE}/v2/front", timeout=1.0).json()
        gps = requests.get(f"{FRODOBOT_BASE}/data", timeout=1.0).json()
        img = decode_frame(cam["front_frame"])
        lat, lon = gps["latitude"], gps["longitude"]
        # LogoNav_frodobot.py와 동일한 부호 규약: orientation(시계방향, deg) → CCW radian
        heading_rad = -float(gps["orientation"]) / 180.0 * math.pi
        self.state.update(camera_img=img, lat=lat, lon=lon,
                           heading_deg=math.degrees(heading_rad))
        return img, lat, lon, heading_rad

    def send_control(self, linear, angular):
        requests.post(f"{FRODOBOT_BASE}/control",
                       data={"command": {"linear": linear, "angular": angular}}, timeout=1.0)

    def maybe_update_frame_buffer(self, img):
        now = time.time()
        if now - self.last_frame_time >= CTX_STRIDE_SEC or len(self.frame_buffer) == 0:
            self.frame_buffer.append(self.obs_transform(img))
            self.last_frame_time = now
            return True
        return False

    def build_inputs(self, lat, lon, heading_rad):
        # 경로는 배포 시작 시 1번만 계산 (osmnav 구조 참고 — 매 프레임 OSRM 재쿼리 안 함)
        if not self._route_initialized:
            self.map_builder.set_goal(lat, lon, self.goal_lat, self.goal_lon)
            self._route_initialized = True
        elif self.map_builder.is_off_route(lat, lon, threshold_m=15.0):
            self.state.log("경로 이탈 감지 → 재라우팅")
            self.map_builder.set_goal(lat, lon, self.goal_lat, self.goal_lon)

        # obs_stack: 컨텍스트가 아직 안 찼으면 가장 오래된 프레임으로 패딩
        frames = list(self.frame_buffer)
        while len(frames) < N_CTX + 1:
            frames = [frames[0]] + frames if frames else frames
        obs_stack = torch.cat(frames[-(N_CTX + 1):], dim=0).unsqueeze(0).to(self.device)  # (1,18,96,96)
        obs_cur = obs_stack[:, -3:]

        map_np = self.map_builder.get_map_image(
            lat, lon, heading_rad,
            past_track=list(self.past_track) if self.past_track else None,
        )
        self.state.update(map_img=map_np)
        map_tensor = self.map_builder.transform(Image.fromarray(map_np)).unsqueeze(0).to(self.device)  # (1,3,96,96)

        goal_pose = torch.zeros(1, 4, device=self.device)
        goal_mask = torch.zeros(1, dtype=torch.long, device=self.device)
        feat_text = torch.zeros(1, 512, device=self.device)
        cur_img = F.interpolate(obs_cur, (224, 224), mode="bilinear", align_corners=False)

        return obs_stack, goal_pose, map_tensor, obs_cur, goal_mask, feat_text, cur_img

    def predict_waypoints(self, lat, lon, heading_rad):
        inputs = self.build_inputs(lat, lon, heading_rad)
        with torch.no_grad():
            pred, _, _ = self.model(*inputs)
        pred_xy_m = pred[0, :, :2].detach().cpu().numpy() * METRIC_WAYPOINT_SPACING  # (8,2) ego x=fwd,y=left
        self.state.update(pred_xy_m=pred_xy_m)
        return pred_xy_m

    def waypoint_to_control(self, pred_xy_m, target_step=2):
        """LogoNav_frodobot.py와 동일한 기하학적 변환 (목표 waypoint까지의 각도/거리로
        linear/angular 속도 산출). target_step=2 → 세 번째 예측 지점(약 0.9초 앞)을 조준."""
        x, y = pred_xy_m[target_step]  # x=forward(m), y=left(m)
        EPS = 1e-8
        if abs(x) < EPS and abs(y) < EPS:
            return 0.0, 0.0
        if abs(x) < EPS:
            return 0.0, np.sign(y) * math.pi / (2 * DT)
        linear = x / DT
        angular = math.atan2(y, x) / DT  # y=left이므로 양수 angular=왼쪽 회전(CCW)이 되도록
        return float(np.clip(linear, 0, MAX_V * 2)), float(np.clip(angular, -MAX_W * 2, MAX_W * 2))

    def step(self):
        img, lat, lon, heading_rad = self.poll_frodobot()

        # GPS fix 없음(sentinel 1000) — 지도 자체를 만들 수 없으므로 정지 유지
        if lat == 1000 or lon == 1000:
            self.state.log_error("GPS fix 없음 (lat/lon=1000) — 정지 유지")
            return 0.0, 0.0

        self.maybe_update_frame_buffer(img)
        self.past_track.append((lat, lon))

        if len(self.frame_buffer) < N_CTX + 1:
            self.state.log("context 채우는 중 ... 정지 유지")
            return 0.0, 0.0

        pred_xy_m = self.predict_waypoints(lat, lon, heading_rad)
        linear, angular = self.waypoint_to_control(pred_xy_m)
        linear, angular = clip_control(linear, angular)
        return linear, angular

    def run(self):
        print("[deploy] 시작 — Ctrl+C로 정지")
        try:
            while True:
                t0 = time.time()
                try:
                    linear, angular = self.step()
                except Exception as e:
                    # 추론/네트워크 등 어떤 에러든 로봇은 반드시 정지시키고 대시보드에 기록.
                    # 에러를 삼키고 계속 움직이는 건 안전상 절대 금지 — 여기서 멈추고 재발생시킴.
                    self.state.log_error(f"step() 실패: {e!r} — 정지 명령 전송 후 중단")
                    self.send_control(0.0, 0.0)
                    raise
                self.send_control(linear, angular)
                elapsed = time.time() - t0
                self.state.update(linear=linear, angular=angular,
                                   loop_hz=round(1.0 / max(elapsed, 1e-6), 2))
                print(f"  linear={linear:+.3f} m/s  angular={angular:+.3f} rad/s")
                time.sleep(max(0.0, DT - elapsed))
        except KeyboardInterrupt:
            print("\n[deploy] 정지 요청됨 — 로봇 정지 명령 전송")
            self.send_control(0.0, 0.0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--map_range", type=float, required=True,
                   help="체크포인트를 학습시킨 맵 반경과 반드시 동일해야 함 "
                        "(baseline=25, 12m실험=12, 20m실험=20). 체크포인트마다 다르므로 기본값 없음 — 반드시 명시.")
    p.add_argument("--goal_lat", type=float, required=True)
    p.add_argument("--goal_lon", type=float, required=True)
    p.add_argument("--debug_port", type=int, default=8080,
                   help="모니터링 웹 대시보드 포트 (0이면 비활성화)")
    args = p.parse_args()

    deployer = OmniVLAEdgeDeployment(
        ckpt_path=args.ckpt, map_range_m=args.map_range,
        goal_lat=args.goal_lat, goal_lon=args.goal_lon,
        debug_port=args.debug_port,
    )
    deployer.run()
