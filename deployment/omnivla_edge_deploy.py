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
  - WAYPOINT_STRIDE_SEC: 체크포인트가 학습된 WAYPOINT_STRIDE와 정확히 일치시킬 것.
      위 25/12/20m 체크포인트는 전부 WAYPOINT_STRIDE=3(≈2m horizon) → 0.3으로 설정.
      2026-09 이후 5m-horizon(WAYPOINT_STRIDE=7)으로 재학습한 체크포인트는 0.7 사용.

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
import json
import argparse
import math
from pathlib import Path
from collections import deque
from datetime import datetime

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
CTX_STRIDE_SEC = 0.3          # 컨텍스트 이미지 캡처 간격 (CTX_STRIDE=3 frames @ ~10Hz)
                               # ⚠ WAYPOINT_STRIDE_SEC과는 별개 상수임 — 우연히 값이
                               # 같았을 뿐(둘 다 예전엔 3프레임), 바꿀 때 혼동하지 말 것.
WAYPOINT_STRIDE_SEC = 0.7      # waypoint 간 시간 간격 (rides11_dataset.py WAYPOINT_STRIDE=7
                               # frames @ ~10Hz). 체크포인트가 어느 WAYPOINT_STRIDE로
                               # 학습됐는지에 정확히 맞춰야 함 (예전 25/12/20m 체크포인트는
                               # WAYPOINT_STRIDE=3 → 이 값 0.3으로 되돌려서 추론할 것).
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
DT    = 1.0 / 3.0  # 제어 루프 주기 (3Hz) — waypoint_to_control()에서는 안 씀(2026-08 버그 수정, 아래 참고)

# 2026-09-18: 추론 시작 전 GPS 궤적부터 확보하는 직진 웜업 단계를 넣었다가 제거함 —
# 로봇이 그 자리에서 실제로 순이동을 못 하는 지점(장애물/노면 등)에서는 웜업이 영원히
# 안 끝나 추론 자체가 시작조차 안 되는 문제가 실측됨(deploy_20260918_1823/1832.jsonl,
# 8초 넘게 GPS 위치가 거의 그대로였음). 대신 아래 EMA 유지 방식으로, 추론은 즉시
# 시작하되(초반 몇 틱은 IMU 폴백) GPS-heading이 한 번이라도 잡히면 그때부터 안정화.

# 2026-09-18: GPS-heading이 imu_fallback과 번갈아 전환되면서 지도 회전이 틱마다
# 수십~백여도씩 튀는 문제 실측 확인(cmp_imu_-214 vs cmp_gps_-147.7 렌더링 비교).
# 그래서 GPS-heading을 "즉시 대체"가 아니라 EMA로 완만하게만 반영하고, GPS 이동량이
# 잠깐 부족해도(제자리 회전/정지 등) 마지막 EMA 값을 그대로 유지("관성")하도록 바꿈 —
# 순간적인 IMU 폴백으로 스냅되지 않게. ALPHA가 작을수록 더 안정적이지만 실제 방향
# 전환에 더 느리게 반응함.
HEADING_EMA_ALPHA = 0.3

# 2026-09-18: is_off_route() 임계값을 15m→3m로 낮췄더니, OSRM이 자체적으로 요청 좌표를
# 가장 가까운 매핑된 길(way)로 "스냅"하는 거리가 그보다 큰 지점(예: 매핑된 보행로가 없는
# 개활지, 실측 3.38m)에서는 재라우팅을 해도 새 경로 시작점이 여전히 3m 넘게 떨어져 있어
# is_off_route()가 계속 True → 재라우팅을 무한 반복하는 폭주가 실측됨
# (deploy_20260918_184622.jsonl, 0.3~0.5초마다 reroute 이벤트). 원인(매핑 안 된 지점에서
# 출발) 자체는 임계값 튜닝으로 못 고치지만, 적어도 매 틱 재쿼리하는 폭주는 쿨다운으로 막는다.
REROUTE_COOLDOWN_S = 3.0

# ── GPS/IMU 자이로 융합 (실험적, 기본 꺼짐) ──────────────────────────────────
# GPS-heading을 못 구하는 구간(gps_ema_hold)에서 지금은 마지막 EMA 값을 그냥
# 고정해서 쓰는데, 그 사이에 로봇이 실제로 방향을 틀면 반영이 안 됨. 대신
# frodobot_raw["gyros"](수직축 각속도)를 적분해서 "그동안 얼마나 돌았는지"만큼
# EMA heading을 보정하는 방식 — IMU 컴퍼스의 "절대값"은 신뢰 안 하고(자기장 간섭으로
# 세션마다 오차가 +11°~+169°까지 들쭉날쭉했음, 고정 오프셋이 아님) "상대 회전량"만
# 신뢰하는 접근. 축/부호/단위(deg/s 가정)를 실측으로 아직 검증 안 했으므로 기본은
# OFF — 다음 실배포에서 켜보고 로그의 gps_ema_gyro 구간 궤적이 실제와 맞는지 확인
# 후 계속 켤지 결정할 것.
USE_GYRO_FUSION = False
GYRO_YAW_AXIS_SIGN = 1.0   # 부호가 반대로 나오면 -1.0으로 뒤집을 것
GYRO_UNIT_IS_DEG = True    # frodobot_raw["gyros"] 값이 deg/s라고 가정 (rad/s면 False로)


def estimate_yaw_delta_from_gyro(raw_data, dt_s):
    """자이로 수직축(z, gravity와 같은 축 — accels z≈1g로 확인됨) 평균 각속도로
    dt_s 동안의 heading 변화량(rad)을 추정. USE_GYRO_FUSION 실험 전용, 검증 전."""
    gyros = raw_data.get("gyros")
    if not gyros or dt_s <= 0:
        return 0.0
    mean_z = sum(g[2] for g in gyros) / len(gyros)
    if GYRO_UNIT_IS_DEG:
        mean_z = math.radians(mean_z)
    return GYRO_YAW_AXIS_SIGN * mean_z * dt_s

# heading 변환: 원래 -orientation/180*pi 하나만 썼는데(90도 보정이 빠져있어 지도가
# 어긋나 보일 수 있다는 가설이 있었음), 실기기 테스트 결과 "얼마나 어긋났는지"를
# 추측으로 고치기보다 GPS 궤적 기반 heading(estimate_heading_from_track, 학습 데이터
# heading과 동일한 산출 방식)과 직접 비교해서 진단하는 쪽으로 방향을 잡음 — step()의
# heading_diff_deg 로그 참고. 여기서 고정 오프셋으로 성급하게 "고치지" 않는다.


def decode_frame(b64_str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


LAT_M = 111320.0  # 위도 1도당 미터 (근거리 근사)
MIN_NET_DISP_M = 0.3  # "거의 정지"만 걸러내는 순변위 하한 (min_disp_m보다 훨씬 작음)


def estimate_heading_from_track(past_track, min_disp_m=1.5):
    """로봇이 실제로 지나온 GPS 궤적(past_track)에서 진행방향을 추정.
    osm_map_generator_rides11.py::estimate_headings()와 동일한 공식(atan2(북쪽성분, 동쪽성분),
    East=0/North=+90 CCW) — 학습 데이터의 heading이 바로 이 방식으로 만들어졌음.

    직전 두 점만 비교하지 않는다 — GPS가 ~0.42m 단위로만 갱신되는 양자화 잡음 때문에,
    인접한 두 fix만 보면 위도/경도 중 어느 쪽이 먼저 양자화 경계를 넘었는지에 따라
    heading이 순간적으로 남↔서로 튀는 문제가 실측으로 확인됨(docs/0825.md 1-1, 원래는
    시각화 스크립트에서 발견·수정된 문제). 그 수정과 동일하게, 누적 이동거리가
    min_disp_m 이상 되는 지점까지 거슬러 올라가 그 구간 전체의 변위로 추정해서
    양자화 스텝 하나짜리 잡음을 평균화한다. 이동량이 부족하면(정지/막 시작) None.

    2026-09-18: 0.8m로 낮췄다가 원복함 — 순이동이 작고 지그재그인 구간에서 0.8m는 노이즈를
    못 걸러내 GPS-heading이 엉뚱한 방향(최대 169° 차이)으로 튀는 부작용이 실측 확인됨
    (deploy_20260918_180335.jsonl). 1.5m가 더 안정적이라 원래 값으로 복귀.

    2026-09-18(2차): 마지막 체크가 "시작-끝 순변위(직선거리) >= min_disp_m"였는데, 로봇이
    완전 직진이 아니라 약간 지그재그(GPS 위경도 축별 비동기 양자화 + 실제 약간의 굴곡)로
    움직이면 누적 경로 길이(cum_m)는 min_disp_m을 넘겨도 순변위는 계속 그보다 작게 남아
    30초 넘게 계속 None만 반환하는 문제가 실측됨 — 그 사이 IMU가 30°+ 드리프트해서 지도가
    계속 잘못된 방향으로 그려짐. cum_m으로 이미 "양자화 잡음 평균화" 조건(min_disp_m 이상
    경로를 거슬러 올라감)을 충족했다면, 마지막 체크는 "순변위가 거의 0인 진짜 정지 상태"만
    걸러내면 되므로 훨씬 작은 MIN_NET_DISP_M로 완화. cum_m이 min_disp_m에 못 미친(이력
    자체가 부족한) 경우엔 기존처럼 min_disp_m 그대로 요구."""
    if len(past_track) < 2:
        return None
    lat_end, lon_end = past_track[-1]
    lat_start, lon_start = past_track[-2]
    cum_m = 0.0
    reached_min_path = False
    for i in range(len(past_track) - 2, -1, -1):
        lat_a, lon_a = past_track[i]
        lat_b, lon_b = past_track[i + 1]
        dlat = (lat_b - lat_a) * LAT_M
        dlon = (lon_b - lon_a) * LAT_M * math.cos(math.radians(lat_a))
        cum_m += math.hypot(dlat, dlon)
        lat_start, lon_start = lat_a, lon_a
        if cum_m >= min_disp_m:
            reached_min_path = True
            break
    dlat = (lat_end - lat_start) * LAT_M
    dlon = (lon_end - lon_start) * LAT_M * math.cos(math.radians(lat_start))
    net_disp_m = math.hypot(dlat, dlon)
    required_m = MIN_NET_DISP_M if reached_min_path else min_disp_m
    if net_disp_m < required_m:
        return None
    return math.atan2(dlat, dlon)


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

        # GPS-heading EMA 상태 (단위원 위 복소수로 유지 — 각도는 선형평균하면 안 되고
        # wraparound(-180/+180 경계)를 다뤄야 하므로 벡터로 평균낸 뒤 각도를 복원함)
        self._heading_ema_vec = None

        # 마지막 재라우팅 시각 (REROUTE_COOLDOWN_S 참고 — 무한 재라우팅 스팸 방지)
        self._last_reroute_ts = 0.0

        # 직전 step()의 record ts (USE_GYRO_FUSION 적분용 dt 계산)
        self._prev_step_ts = None

        # ── 데이터분석용 로그 (JSONL, 실행마다 날짜시간별 파일) ──
        log_dir = REPO_ROOT / "deployment" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = log_dir / f"deploy_{run_id}.jsonl"
        self._log_fp = open(self.log_path, "a", buffering=1, encoding="utf-8")
        print(f"[deploy] 로그 저장 경로: {self.log_path}")
        self._log_jsonl({
            "type": "run_start", "ts": time.time(), "run_id": run_id,
            "ckpt_path": str(ckpt_path), "map_range_m": map_range_m,
            "goal_lat": goal_lat, "goal_lon": goal_lon,
        })

    def _log_jsonl(self, record: dict):
        self._log_fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    def poll_frodobot(self):
        cam = requests.get(f"{FRODOBOT_BASE}/v2/front", timeout=5.0).json()
        gps = requests.get(f"{FRODOBOT_BASE}/data", timeout=5.0).json()
        img = decode_frame(cam["front_frame"])
        lat, lon = gps["latitude"], gps["longitude"]
        orientation_deg_raw = float(gps["orientation"])
        # LogoNav_frodobot.py와 동일한 부호 규약: orientation(시계방향, deg) → CCW radian.
        # 90도 보정이 빠져 있다는 가설이 있었으나, step()의 IMU-vs-GPS heading 비교
        # 로그로 먼저 진단하기로 함 — 여기서 성급하게 상수를 바꾸지 않음.
        heading_rad = -float(orientation_deg_raw) / 180.0 * math.pi
        heading_deg = math.degrees(heading_rad)
        self.state.update(camera_img=img, lat=lat, lon=lon,
                           heading_deg=heading_deg, orientation_deg_raw=orientation_deg_raw)
        return img, lat, lon, heading_rad, gps

    def send_control(self, linear, angular):
        # 전송 실패 시 예외를 그대로 올려서 run()의 루프가 멈추고 정지 명령이 강제되도록 함
        # (에러를 삼키고 계속 움직이는 건 안전상 절대 금지 — step()과 동일한 원칙).
        r = requests.post(f"{FRODOBOT_BASE}/control",
                           json={"command": {"linear": linear, "angular": angular}}, timeout=5.0)
        return r.status_code

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
            self._log_jsonl({"type": "event", "ts": time.time(), "event": "route_init",
                              "lat": lat, "lon": lon,
                              "route_latlon": self.map_builder.get_route_latlon()})
        elif (self.map_builder.is_off_route(lat, lon, threshold_m=3.0)
                and time.time() - self._last_reroute_ts >= REROUTE_COOLDOWN_S):
            self.state.log("경로 이탈 감지 → 재라우팅")
            self.map_builder.set_goal(lat, lon, self.goal_lat, self.goal_lon)
            self._last_reroute_ts = time.time()
            self._log_jsonl({"type": "event", "ts": time.time(), "event": "reroute",
                              "lat": lat, "lon": lon,
                              "route_latlon": self.map_builder.get_route_latlon()})

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
        self._last_map_img = map_np  # predict_waypoints()에서 예측 궤적을 겹쳐 그리는 데 사용
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
        # 예측 궤적(청록색)을 방금 만든 ego-centric 지도(빨강=계획 경로, 회색=지나온 길) 위에
        # 같은 축척으로 겹쳐서 대시보드에 표시 — 모델이 실제 경로 대비 어디를 보고 있는지 한눈에 확인용.
        overlay_img = self.map_builder.draw_predicted_trajectory(self._last_map_img, pred_xy_m)
        self.state.update(map_img=overlay_img)
        return pred_xy_m

    def waypoint_to_control(self, pred_xy_m, target_step=2):
        """LogoNav_frodobot.py와 동일한 기하학적 변환(목표 waypoint까지의 각도/거리로
        linear/angular 속도 산출)이되, 시간 분모는 반드시 그 waypoint의 실제 시점과
        일치시켜야 함. LogoNav는 DT=1/4를 쓰는데 이건 "NoMaD 모델 자체의 웨이포인트
        간격이 0.25초"이기 때문에 맞는 값이었음 — 우리 모델의 웨이포인트 간격은
        0.7초(WAYPOINT_STRIDE_SEC)이고, index i(0-based)는 (i+1)*0.7초 뒤를 의미함
        (rides11_dataset.py의 k=range(1, N_WAYPOINTS+1) 인덱싱과 동일).
        이걸 제어 루프 주기 DT(=1/3)로 나누면 시점이 안 맞아서 속도가 실제보다
        부풀려지고, 그 결과 항상 MAX_V/MAX_W 안전 캡에 걸려 모델 예측의 크기 정보가
        사라지는 문제가 있었음 (2026-08 실배포 테스트에서 "명령이 항상 작다"로 발견됨)."""
        target_time_s = (target_step + 1) * WAYPOINT_STRIDE_SEC  # 예: target_step=2 → 2.1초
        x, y = pred_xy_m[target_step]  # x=forward(m), y=left(m)
        EPS = 1e-8
        if abs(x) < EPS and abs(y) < EPS:
            return 0.0, 0.0
        if abs(x) < EPS:
            return 0.0, np.sign(y) * math.pi / (2 * target_time_s)
        linear = x / target_time_s
        angular = math.atan2(y, x) / target_time_s  # y=left이므로 양수 angular=왼쪽 회전(CCW)이 되도록
        return float(np.clip(linear, 0, MAX_V * 2)), float(np.clip(angular, -MAX_W * 2, MAX_W * 2))

    def step(self):
        img, lat, lon, heading_rad, raw_data = self.poll_frodobot()
        imu_deg = math.degrees(heading_rad)
        # frodobot_raw: FrodoBot Mini가 /data로 내보내는 원본 텔레메트리 그대로 보존
        # (battery, signal_level, speed, gps_signal, vibration, accels/gyros/mags/rpms 등).
        record = {"type": "step", "ts": time.time(),
                  "lat": lat, "lon": lon, "imu_heading_deg": imu_deg,
                  "frodobot_raw": raw_data}

        # GPS fix 없음(sentinel 1000) — 지도 자체를 만들 수 없으므로 정지 유지
        if lat == 1000 or lon == 1000:
            self.state.log_error("GPS fix 없음 (lat/lon=1000) — 정지 유지")
            record.update(gps_fix_ok=False, linear=0.0, angular=0.0, note="gps_fix_missing")
            self._log_jsonl(record)
            return 0.0, 0.0
        record["gps_fix_ok"] = True

        self.maybe_update_frame_buffer(img)
        self.past_track.append((lat, lon))

        # heading 소스: 2026-08-25 실배포 로그 분석(docs/0825.md 2-2)에서 IMU 컴퍼스와
        # GPS궤적 기반 heading이 평균 +97° 어긋남을 확인 — 학습 데이터의 heading은
        # GPS궤적 기반(atan2, osm_map_generator_rides11.py::estimate_headings()와 동일
        # 공식)으로 만들어지므로, render_frame()에 넣는 heading도 같은 소스로 맞춰서
        # 학습/배포 간 heading 정의 자체가 갈라지는 것을 원천 차단한다.
        #   imu_heading  = -orientation/180*pi (로봇 컴퍼스 센서, 진단/폴백 전용)
        #   gps_heading  = 방금 지나온 GPS 두 점 사이 방향 (학습 heading과 동일 산출 방식)
        # GPS 이동량이 부족(정지/막 시작)해서 gps_heading을 못 구할 때만 IMU로 폴백.
        gps_heading_rad = estimate_heading_from_track(list(self.past_track))
        record["gps_heading_deg"] = math.degrees(gps_heading_rad) if gps_heading_rad is not None else None

        if gps_heading_rad is not None:
            new_vec = complex(math.cos(gps_heading_rad), math.sin(gps_heading_rad))
            if self._heading_ema_vec is None:
                self._heading_ema_vec = new_vec  # 첫 확보 시엔 그대로 초기화
            else:
                self._heading_ema_vec = (1 - HEADING_EMA_ALPHA) * self._heading_ema_vec + HEADING_EMA_ALPHA * new_vec
            map_heading_rad = math.atan2(self._heading_ema_vec.imag, self._heading_ema_vec.real)
            smoothed_deg = math.degrees(map_heading_rad)
            diff = (smoothed_deg - imu_deg + 180) % 360 - 180
            record["heading_diff_deg"] = diff
            record["smoothed_heading_deg"] = smoothed_deg
            record["map_heading_source"] = "gps_track"
            print(f"    [heading] IMU컴퍼스={imu_deg:+7.1f}°  GPS궤적={record['gps_heading_deg']:+7.1f}°  "
                  f"EMA사용={smoothed_deg:+7.1f}°  차이(EMA-IMU)={diff:+7.1f}°")
        elif self._heading_ema_vec is not None:
            # GPS 이동량이 잠깐 부족해도(제자리 회전/정지) 순간 IMU로 스냅하지 않고
            # 마지막으로 안정화된 EMA heading을 유지("관성") — 2026-09-18에 틱마다
            # 최대 190°까지 튀던 회전 불안정을 직접 렌더링 비교로 확인해서 도입.
            # USE_GYRO_FUSION=True면 그냥 고정하지 않고, 자이로 상대 회전량만큼
            # 계속 보정한다(estimate_yaw_delta_from_gyro 참고, 실험적).
            if USE_GYRO_FUSION and self._prev_step_ts is not None:
                dt_s = record["ts"] - self._prev_step_ts
                yaw_delta_rad = estimate_yaw_delta_from_gyro(raw_data, dt_s)
                cur_rad = math.atan2(self._heading_ema_vec.imag, self._heading_ema_vec.real)
                new_rad = cur_rad + yaw_delta_rad
                self._heading_ema_vec = complex(math.cos(new_rad), math.sin(new_rad))
                source_label = "gps_ema_gyro"
            else:
                source_label = "gps_ema_hold"
            map_heading_rad = math.atan2(self._heading_ema_vec.imag, self._heading_ema_vec.real)
            smoothed_deg = math.degrees(map_heading_rad)
            record["heading_diff_deg"] = None
            record["smoothed_heading_deg"] = smoothed_deg
            record["map_heading_source"] = source_label
            print(f"    [heading] IMU컴퍼스={imu_deg:+7.1f}°(무시)  GPS궤적=(이동량 부족)  "
                  f"EMA유지({source_label})={smoothed_deg:+7.1f}°")
        else:
            map_heading_rad = heading_rad  # 콜드스타트 폴백: 아직 GPS-heading을 한 번도 못 구한 초반 몇 틱만 해당
            record["heading_diff_deg"] = None
            record["smoothed_heading_deg"] = None
            record["map_heading_source"] = "imu_fallback"
            print(f"    [heading] IMU컴퍼스={imu_deg:+7.1f}°(폴백 사용, EMA 없음)  GPS궤적=(이동량 부족, 추정불가)")

        self._prev_step_ts = record["ts"]  # USE_GYRO_FUSION dt 계산용

        if len(self.frame_buffer) < N_CTX + 1:
            self.state.log("context 채우는 중 ... 정지 유지")
            record.update(linear=0.0, angular=0.0, note="context_filling")
            self._log_jsonl(record)
            return 0.0, 0.0

        pred_xy_m = self.predict_waypoints(lat, lon, map_heading_rad)
        linear, angular = self.waypoint_to_control(pred_xy_m)
        linear, angular = clip_control(linear, angular)
        record.update(linear=linear, angular=angular,
                       pred_xy_m=pred_xy_m.tolist())
        self._log_jsonl(record)
        return linear, angular

    def run(self):
        # SDK 서버의 헤드리스 브라우저(pyppeteer)는 첫 요청에서 Chrome 실행 + 페이지 접속 +
        # RTM join까지 해서 수 초~십수 초가 걸림. 실시간 루프(1s 타임아웃) 안에서 이 콜드스타트를
        # 맞으면 ReadTimeout으로 죽으므로, 루프 시작 전에 넉넉한 타임아웃으로 미리 깨워둔다.
        print("[deploy] SDK 서버 워밍업 중 (헤드리스 브라우저 초기화 대기)...")
        requests.get(f"{FRODOBOT_BASE}/data", timeout=30.0)
        # 2026-09-18: /data(GPS/텔레메트리)는 준비됐는데 카메라 RTM 채널은 아직 join 중이라
        # /v2/front가 404("Front frame not available")를 반환하는 경우가 실측됨 — poll_frodobot()이
        # 이걸 그대로 .json()해서 front_frame 없는 dict를 받아 KeyError로 죽었음. 카메라도 따로
        # 준비될 때까지 재시도.
        for attempt in range(30):
            r = requests.get(f"{FRODOBOT_BASE}/v2/front", timeout=5.0)
            if r.status_code == 200 and "front_frame" in r.json():
                break
            time.sleep(1.0)
        else:
            raise RuntimeError("카메라 스트림이 30초 안에 준비되지 않음 — SDK 서버/카메라 연결 확인 필요")
        print("[deploy] 워밍업 완료")

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
                    self._log_jsonl({"type": "event", "ts": time.time(),
                                      "event": "step_failed", "error": repr(e)})
                    self.send_control(0.0, 0.0)
                    raise
                control_status = self.send_control(linear, angular)
                self._log_jsonl({"type": "control_sent", "ts": time.time(),
                                  "linear": linear, "angular": angular,
                                  "http_status": control_status})
                elapsed = time.time() - t0
                loop_hz = round(1.0 / max(elapsed, 1e-6), 2)
                self.state.update(linear=linear, angular=angular, loop_hz=loop_hz)
                print(f"  linear={linear:+.3f} m/s  angular={angular:+.3f} rad/s")
                time.sleep(max(0.0, DT - elapsed))
        except KeyboardInterrupt:
            print("\n[deploy] 정지 요청됨 — 로봇 정지 명령 전송")
            self.send_control(0.0, 0.0)
            self._log_jsonl({"type": "event", "ts": time.time(), "event": "run_end",
                              "reason": "keyboard_interrupt"})
        finally:
            print(f"[deploy] 로그 저장 완료: {self.log_path}")
            self._log_fp.close()


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
