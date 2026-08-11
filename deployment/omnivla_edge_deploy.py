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
DT    = 1.0 / 3.0  # 제어 루프 주기 (3Hz) — waypoint_to_control()에서는 안 씀(2026-08 버그 수정, 아래 참고)

# heading 변환: 원래 -orientation/180*pi 하나만 썼는데(90도 보정이 빠져있어 지도가
# 어긋나 보일 수 있다는 가설이 있었음), 실기기 테스트 결과 "얼마나 어긋났는지"를
# 추측으로 고치기보다 GPS 궤적 기반 heading(estimate_heading_from_track, 학습 데이터
# heading과 동일한 산출 방식)과 직접 비교해서 진단하는 쪽으로 방향을 잡음 — step()의
# heading_diff_deg 로그 참고. 여기서 고정 오프셋으로 성급하게 "고치지" 않는다.


def decode_frame(b64_str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


LAT_M = 111320.0  # 위도 1도당 미터 (근거리 근사)


def estimate_heading_from_track(past_track, min_disp_m=0.3):
    """로봇이 실제로 지나온 GPS 궤적(past_track) 최근 두 점으로 진행방향을 추정.
    osm_map_generator_rides11.py::estimate_headings()와 동일한 공식(atan2(북쪽성분, 동쪽성분),
    East=0/North=+90 CCW) — 학습 데이터의 heading이 바로 이 방식으로 만들어졌음.
    이동량이 min_disp_m보다 작으면(정지/GPS 지터) None 반환."""
    if len(past_track) < 2:
        return None
    lat1, lon1 = past_track[-2]
    lat2, lon2 = past_track[-1]
    dlat = (lat2 - lat1) * LAT_M
    dlon = (lon2 - lon1) * LAT_M * math.cos(math.radians(lat1))
    if math.hypot(dlat, dlon) < min_disp_m:
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
        elif self.map_builder.is_off_route(lat, lon, threshold_m=15.0):
            self.state.log("경로 이탈 감지 → 재라우팅")
            self.map_builder.set_goal(lat, lon, self.goal_lat, self.goal_lon)
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
        0.3초(CTX_STRIDE_SEC)이고, index i(0-based)는 (i+1)*0.3초 뒤를 의미함
        (rides11_dataset.py의 k=range(1, N_WAYPOINTS+1) 인덱싱과 동일).
        이걸 제어 루프 주기 DT(=1/3)로 나누면 시점이 안 맞아서 속도가 실제보다
        부풀려지고, 그 결과 항상 MAX_V/MAX_W 안전 캡에 걸려 모델 예측의 크기 정보가
        사라지는 문제가 있었음 (2026-08 실배포 테스트에서 "명령이 항상 작다"로 발견됨)."""
        target_time_s = (target_step + 1) * CTX_STRIDE_SEC  # 예: target_step=2 → 0.9초
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

        # [진단용] IMU(컴퍼스) heading vs GPS 궤적 기반 heading 비교.
        #   imu_heading  = -orientation/180*pi (로봇 컴퍼스 센서, 현재 배포 코드가 실제로 쓰는 값)
        #   gps_heading  = 방금 지나온 GPS 두 점 사이 방향, atan2 (학습 데이터 heading과 동일 방식)
        # 두 값이 계속 크게 어긋나면 컴퍼스 heading이 학습 때 heading과 안 맞는다는 뜻 —
        # render_frame()에 들어가는 heading이 부정확해서 지도가 heading-up으로 안 맞을 수 있음.
        gps_heading_rad = estimate_heading_from_track(list(self.past_track))
        if gps_heading_rad is not None:
            gps_deg = math.degrees(gps_heading_rad)
            diff = (gps_deg - imu_deg + 180) % 360 - 180
            record["gps_heading_deg"] = gps_deg
            record["heading_diff_deg"] = diff
            print(f"    [heading] IMU컴퍼스={imu_deg:+7.1f}°  GPS궤적={gps_deg:+7.1f}°  차이={diff:+7.1f}°")
        else:
            record["gps_heading_deg"] = None
            record["heading_diff_deg"] = None
            print(f"    [heading] IMU컴퍼스={imu_deg:+7.1f}°  GPS궤적=(이동량 부족, 추정불가)")

        if len(self.frame_buffer) < N_CTX + 1:
            self.state.log("context 채우는 중 ... 정지 유지")
            record.update(linear=0.0, angular=0.0, note="context_filling")
            self._log_jsonl(record)
            return 0.0, 0.0

        pred_xy_m = self.predict_waypoints(lat, lon, heading_rad)
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
