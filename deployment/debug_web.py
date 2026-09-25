"""
debug_web.py

배포 중 실시간 모니터링용 웹 대시보드. FrodoBot SDK 자체 페이지(/sdk)는 카메라
영상만 보여주고 우리 파이프라인(지도 생성, GPS, 모델 예측, 에러)은 안 보여주므로
별도로 띄운다. 외부 의존성 없이 표준 라이브러리(http.server)만 사용.

사용법 (omnivla_edge_deploy.py에서 통합 사용):
    from debug_web import DeploymentState, start_debug_server
    state = DeploymentState()
    start_debug_server(state, port=8080)
    ...
    state.update(camera_img=img, map_img=map_np, lat=lat, lon=lon,
                 heading_deg=..., linear=..., angular=..., pred_xy_m=...)
    state.log("경로 이탈 감지 → 재라우팅")
    state.log_error("OSRM 쿼리 실패: ...")

브라우저에서 http://<이 머신 IP>:8080 접속 (1초마다 자동 새로고침).
"""

import io
import json
import queue
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image, ImageDraw

_UNSET = object()  # update()에서 "이 호출은 이 필드를 안 건드림"과 "명시적으로 None"을 구분하기 위한 센티널

# 예측 궤적 확대 패널 전용 스케일. 지도(map.jpg)는 MAP_RANGE_M(예: 20m, 전체 40m)
# 축척 그대로라 8-step 예측(~2m)이 전체 폭의 5%(≈11px)밖에 안 돼서 육안으로 방향
# 판단이 사실상 불가능함 — 그래서 지도와 별개로, 예측 궤적만 이 좁은 고정 범위로
# 확대해서 그리는 패널을 추가함 (모델/제어 로직과 무관, 순수 시각화용).
TRAJ_ZOOM_SIZE = 224
TRAJ_ZOOM_LATERAL_M = 1.2   # 좌우 ±1.2m
TRAJ_ZOOM_FORWARD_M = 2.2   # 전방 0~2.2m (ego는 하단 근처)


class DeploymentState:
    """스레드 세이프한 최신 배포 상태 저장소. 제어 루프가 매 tick 업데이트하고,
    웹 서버 스레드가 읽어서 응답한다."""

    def __init__(self, log_maxlen=100):
        self._lock = threading.Lock()
        self.camera_img = None      # PIL.Image
        self.map_img = None         # PIL.Image
        self.lat = None
        self.lon = None
        self.heading_deg = None
        self.orientation_deg_raw = None
        self.linear = 0.0           # 실제로 로봇에 전송된 값 (dry_run이면 항상 0)
        self.angular = 0.0
        self.computed_linear = 0.0  # waypoint_to_control()이 계산한 값 (dry_run 여부와 무관)
        self.computed_angular = 0.0
        self.pred_xy_m = None       # (8,2) numpy, 예측 웨이포인트
        self.last_update_ts = None
        self.loop_hz = None
        # 2026-09-25 추가: GPS/heading/제어 파이프라인 신뢰도를 GO 누르기 전에
        # 눈으로 확인하기 위한 필드들 (7번 dry-run 게이트와 짝을 이룸)
        self.fix_quality = None
        self.gps_data_ts = None          # 로봇이 보고한 원본 timestamp (staleness 판단용)
        self.gps_heading_deg = None      # GPS 궤적 기반 raw heading (EMA 적용 전)
        self.route_bearing_deg = None    # 경로를 따라가려면 가야 할 방향
        self.heading_route_diff_deg = None  # 최종 heading - route_bearing (클수록 지도가 엉뚱한 방향)
        self.map_rotation_deg = None     # render_frame()에 실제로 넘어간 회전각(90-heading_deg)
        self.target_waypoint_xy = None   # waypoint_to_control()이 조준한 (x,y) 원본 미터값
        self.control_latency_ms = None   # 직전 /control 왕복시간
        self.tick_id = None
        self.dry_run = False
        # 2026-09-25 추가: route-aligned initial heading bootstrap + ARM/GO LIVE
        # 2단계 확인 워크플로용 필드 (설계 근거는 omnivla_edge_deploy.py 참고)
        self.control_stage = "DRY_RUN"   # "DRY_RUN" / "ARMED" / "LIVE"
        self.map_heading_source = None   # "route_aligned" / "gps_track" / "gps_ema_hold" / "gps_ema_gyro" / "imu_fallback"
        self.route_aligned_heading_deg = None
        self.gps_accumulated_path_m = None
        self.gps_net_displacement_m = None
        self.osrm_fallback = None
        self.start_snap_m = None
        self.goal_snap_m = None
        self.map_img_northup = None      # PIL.Image, heading-up 회전 전 미리보기
        self._logs = deque(maxlen=log_maxlen)
        self._errors = deque(maxlen=log_maxlen)

    def update(self, camera_img=None, map_img=None, lat=None, lon=None,
               heading_deg=None, linear=None, angular=None, pred_xy_m=None,
               loop_hz=None, orientation_deg_raw=None, fix_quality=None,
               gps_data_ts=None, gps_heading_deg=_UNSET, route_bearing_deg=_UNSET,
               heading_route_diff_deg=_UNSET, map_rotation_deg=None,
               target_waypoint_xy=None, control_latency_ms=None, tick_id=None,
               dry_run=None, computed_linear=None, computed_angular=None,
               control_stage=None, map_heading_source=_UNSET,
               route_aligned_heading_deg=_UNSET, gps_accumulated_path_m=None,
               gps_net_displacement_m=None, osrm_fallback=_UNSET,
               start_snap_m=_UNSET, goal_snap_m=_UNSET, map_img_northup=None):
        # gps_heading_deg/route_bearing_deg/heading_route_diff_deg는 "이번 틱에
        # 못 구했다"는 의미로 명시적 None이 넘어올 수 있어서, 기본값을 _UNSET으로
        # 두고 "호출에서 아예 안 건드린 경우"와 구분한다 — 그냥 None 기본값을 쓰면
        # poll_frodobot()처럼 이 필드들을 모르는 다른 update() 호출이 매 틱마다
        # 값을 None으로 지워버리게 됨.
        with self._lock:
            if camera_img is not None: self.camera_img = camera_img
            if map_img is not None: self.map_img = map_img
            if lat is not None: self.lat = lat
            if lon is not None: self.lon = lon
            if heading_deg is not None: self.heading_deg = heading_deg
            if orientation_deg_raw is not None: self.orientation_deg_raw = orientation_deg_raw
            if linear is not None: self.linear = linear
            if computed_linear is not None: self.computed_linear = computed_linear
            if computed_angular is not None: self.computed_angular = computed_angular
            if angular is not None: self.angular = angular
            if pred_xy_m is not None: self.pred_xy_m = pred_xy_m
            if loop_hz is not None: self.loop_hz = loop_hz
            if fix_quality is not None: self.fix_quality = fix_quality
            if gps_data_ts is not None: self.gps_data_ts = gps_data_ts
            if gps_heading_deg is not _UNSET: self.gps_heading_deg = gps_heading_deg
            if route_bearing_deg is not _UNSET: self.route_bearing_deg = route_bearing_deg
            if heading_route_diff_deg is not _UNSET: self.heading_route_diff_deg = heading_route_diff_deg
            if map_rotation_deg is not None: self.map_rotation_deg = map_rotation_deg
            if target_waypoint_xy is not None: self.target_waypoint_xy = target_waypoint_xy
            if control_latency_ms is not None: self.control_latency_ms = control_latency_ms
            if tick_id is not None: self.tick_id = tick_id
            if dry_run is not None: self.dry_run = dry_run
            if control_stage is not None: self.control_stage = control_stage
            if map_heading_source is not _UNSET: self.map_heading_source = map_heading_source
            if route_aligned_heading_deg is not _UNSET: self.route_aligned_heading_deg = route_aligned_heading_deg
            if gps_accumulated_path_m is not None: self.gps_accumulated_path_m = gps_accumulated_path_m
            if gps_net_displacement_m is not None: self.gps_net_displacement_m = gps_net_displacement_m
            if osrm_fallback is not _UNSET: self.osrm_fallback = osrm_fallback
            if start_snap_m is not _UNSET: self.start_snap_m = start_snap_m
            if goal_snap_m is not _UNSET: self.goal_snap_m = goal_snap_m
            if map_img_northup is not None: self.map_img_northup = map_img_northup
            self.last_update_ts = time.time()

    def log(self, msg):
        with self._lock:
            self._logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def log_error(self, msg):
        with self._lock:
            err = f"[{time.strftime('%H:%M:%S')}] ERROR: {msg}"
            self._errors.append(err)
            self._logs.append(err)

    def snapshot_status(self):
        with self._lock:
            age = time.time() - self.last_update_ts if self.last_update_ts else None
            gps_age = None
            if self.gps_data_ts is not None:
                try:
                    gps_age = time.time() - float(self.gps_data_ts)
                except (TypeError, ValueError):
                    gps_age = None
            return {
                "lat": self.lat, "lon": self.lon,
                "heading_deg": self.heading_deg,
                "orientation_deg_raw": self.orientation_deg_raw,
                "linear": self.linear, "angular": self.angular,
                "computed_linear": self.computed_linear, "computed_angular": self.computed_angular,
                "loop_hz": self.loop_hz,
                "age_sec": round(age, 2) if age is not None else None,
                "pred_xy_m": self.pred_xy_m.tolist() if self.pred_xy_m is not None else None,
                "logs": list(self._logs)[-30:],
                "n_errors": len(self._errors),
                "gps_ok": (self.lat is not None and self.lat != 1000 and self.lon != 1000),
                # 2026-09-25 추가 필드 (7번 dry-run GO 체크리스트용)
                "fix_quality": self.fix_quality,
                "gps_age_sec": round(gps_age, 2) if gps_age is not None else None,
                "gps_heading_deg": self.gps_heading_deg,
                "route_bearing_deg": self.route_bearing_deg,
                "heading_route_diff_deg": self.heading_route_diff_deg,
                "map_rotation_deg": self.map_rotation_deg,
                "target_waypoint_xy": list(self.target_waypoint_xy) if self.target_waypoint_xy is not None else None,
                "control_latency_ms": self.control_latency_ms,
                "tick_id": self.tick_id,
                "dry_run": self.dry_run,
                # 2026-09-25 추가 필드
                "control_stage": self.control_stage,
                "map_heading_source": self.map_heading_source,
                "route_aligned_heading_deg": self.route_aligned_heading_deg,
                "gps_accumulated_path_m": self.gps_accumulated_path_m,
                "gps_net_displacement_m": self.gps_net_displacement_m,
                "osrm_fallback": self.osrm_fallback,
                "start_snap_m": self.start_snap_m,
                "goal_snap_m": self.goal_snap_m,
            }

    def snapshot_camera_jpeg(self):
        with self._lock:
            img = self.camera_img
        return _to_jpeg(img)

    def snapshot_map_jpeg(self):
        with self._lock:
            img = self.map_img
        return _to_jpeg(img)

    def snapshot_map_northup_jpeg(self):
        with self._lock:
            img = self.map_img_northup
        return _to_jpeg(img)

    def snapshot_traj_jpeg(self):
        with self._lock:
            pred = self.pred_xy_m
        return _to_jpeg(_render_traj_zoom(pred))


def _render_traj_zoom(pred_xy_m, target_step=2):
    """예측 waypoint(ego frame, x=forward m, y=left m)를 지도 축척과 무관한
    좁은 고정 범위(TRAJ_ZOOM_*)로 확대해서 그린 이미지 반환. 격자선(0.5m 간격)과
    스케일 라벨을 넣어서 지도(map.jpg)보다 훨씬 크게, 방향이 눈에 보이게 함."""
    size = TRAJ_ZOOM_SIZE
    img = Image.new("RGB", (size, size), (18, 18, 18))
    d = ImageDraw.Draw(img)

    lat_m, fwd_m = TRAJ_ZOOM_LATERAL_M, TRAJ_ZOOM_FORWARD_M
    px_per_m = min(size / (2 * lat_m), (size - 30) / fwd_m)
    cx = size / 2.0
    cy = size - 15  # ego를 하단 근처에 둬서 전방으로 뻗어나갈 공간 확보

    def to_px(x, y):  # x=forward(m), y=left(m) → (col, row)
        return (cx - y * px_per_m, cy - x * px_per_m)

    # 격자선 0.5m 간격
    step_px = 0.5 * px_per_m
    x_ = cx
    while x_ < size:
        d.line([(x_, 0), (x_, size)], fill=(40, 40, 40)); x_ += step_px
    x_ = cx - step_px
    while x_ > 0:
        d.line([(x_, 0), (x_, size)], fill=(40, 40, 40)); x_ -= step_px
    y_ = cy
    while y_ > 0:
        d.line([(0, y_), (size, y_)], fill=(40, 40, 40)); y_ -= step_px

    # 중심축(로봇 정면 방향) 강조
    d.line([(cx, 0), (cx, size)], fill=(70, 70, 70))
    d.line([(0, cy), (size, cy)], fill=(70, 70, 70))

    if pred_xy_m is not None and len(pred_xy_m) > 0:
        pts = [to_px(x, y) for x, y in pred_xy_m]
        d.line(pts, fill=(0, 255, 255), width=2)
        for k, p in enumerate(pts):
            r = 5 if k == target_step else 3
            d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r],
                      fill=(0, 255, 255), outline=(255, 255, 255))

    # ego 마커
    ex, ey = to_px(0, 0)
    d.ellipse([ex - 5, ey - 5, ex + 5, ey + 5], fill=(60, 255, 60), outline=(255, 255, 255))
    d.text((4, 2), f"±{lat_m}m / {fwd_m}m  (0.5m 격자)", fill=(180, 180, 180))
    return img


def _to_jpeg(img):
    if img is None:
        img = Image.new("RGB", (224, 224), (40, 40, 40))
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


_PAGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>OmniVLA-Edge 배포 모니터</title>
<style>
body { font-family: monospace; background:#111; color:#eee; margin:20px; }
img { width: 300px; height: 300px; object-fit: contain; background:#000; border:1px solid #444; }
.row { display:flex; gap:20px; flex-wrap:wrap; }
.card { background:#1c1c1c; padding:12px; border-radius:8px; }
table { border-collapse: collapse; }
td { padding: 2px 10px 2px 0; }
.ok { color:#6f6; } .bad { color:#f66; } .warn { color:#fa0; }
pre { background:#000; padding:8px; height:260px; overflow-y:auto; font-size:12px; }
button { font-family: monospace; font-size:14px; padding:8px 16px; margin:4px 6px 4px 0; border-radius:6px;
         border:1px solid #555; cursor:pointer; }
button:disabled { opacity:0.35; cursor:not-allowed; }
#btnConfirm { background:#245; color:#fff; }
#btnArm { background:#640; color:#fff; }
#btnGoLive { background:#600; color:#fff; font-weight:bold; }
#btnAbort { background:#333; color:#fff; }
</style></head>
<body>
<h2>OmniVLA-Edge 실시간 배포 모니터 <span id="stageBadge"></span></h2>
<div class="card" style="margin-bottom:16px;">
  <b>Pre-drive 워크플로</b> — DRY RUN(관찰) → 정렬 확인 → ARM → GO LIVE(사용자 명시 확인 필요)
  <div style="margin-top:8px;">
    <button id="btnConfirm" onclick="doAction('/confirm_alignment', '로봇을 route 방향으로 물리적으로 정렬했습니다. 그 방향을 initial heading으로 사용합니다. 맞습니까?')">1. 정렬 확인 (route-aligned heading 확정)</button>
    <button id="btnArm" onclick="doAction('/arm', 'ARM 하시겠습니까? (아직 실제 명령은 전송되지 않습니다)')">2. ARM</button>
    <button id="btnGoLive" onclick="doAction('/go_live', '⚠ 정말 GO LIVE 하시겠습니까?\\n이 순간부터 실제 로봇에 non-zero 명령이 전송될 수 있습니다.')">3. GO LIVE</button>
    <button id="btnAbort" onclick="doAction('/abort', null)">ABORT → DRY RUN으로 복귀</button>
  </div>
</div>
<div class="row">
  <div class="card"><b>카메라</b><br><img src="/frame.jpg?t=__TS__"></div>
  <div class="card"><b>North-up 경로 미리보기 (회전 전)</b><br><img src="/map_northup.jpg?t=__TS__">
    <div style="font-size:11px; margin-top:4px;">route geometry 자체(스냅/OSRM/보도 위치)가 맞는지 확인용 — 회전 로직 안 들어감</div>
  </div>
  <div class="card"><b>최종 heading-up 모델 입력 지도</b><br><img src="/map.jpg?t=__TS__">
    <div style="font-size:11px; margin-top:4px; line-height:1.6;">
      <span style="color:#f66;">■</span> 계획 경로(OSRM)&nbsp;
      <span style="color:#999;">■</span> 지나온 길&nbsp;
      <span style="color:#3f3;">●</span> 현재 위치&nbsp;
      <span style="color:#0ff;">■</span> 모델 예측 궤적&nbsp;
      <span style="color:#fff;">—</span> 스케일바
    </div>
  </div>
  <div class="card"><b>예측 궤적 확대(지도와 별개 축척)</b><br><img src="/traj.jpg?t=__TS__">
    <div style="font-size:11px; margin-top:4px;">지도(40m 등)가 넓어서 실제 예측(~2m)이 안 보이는 문제 때문에 추가된 확대 패널</div>
  </div>
  <div class="card">
    <b>상태 — Localization / Heading</b>
    <table id="statusLoc"><tbody></tbody></table>
  </div>
  <div class="card">
    <b>상태 — Route</b>
    <table id="statusRoute"><tbody></tbody></table>
  </div>
  <div class="card">
    <b>상태 — Model / Control</b>
    <table id="statusModel"><tbody></tbody></table>
  </div>
</div>
<div class="card" style="margin-top:16px;">
  <b>로그 (최근 30줄)</b>
  <pre id="logs"></pre>
</div>
<script>
function fmtNum(v, digits) {
  return (v === null || v === undefined) ? '—' : Number(v).toFixed(digits);
}

async function doAction(path, confirmMsg) {
  if (confirmMsg && !confirm(confirmMsg)) return;
  await fetch(path, {method: 'POST'});
  poll();
}

const SOURCE_LABEL = {
  route_aligned: '사용자 route-align (고정)',
  gps_track: 'GPS 궤적 (실측, EMA)',
  gps_ema_hold: 'GPS EMA 유지(관성)',
  gps_ema_gyro: 'GPS EMA + 자이로 보정',
  imu_fallback: 'IMU 폴백 (비권장)',
};
const SOURCE_CLASS = {
  route_aligned: 'warn', gps_track: 'ok', gps_ema_hold: 'ok',
  gps_ema_gyro: 'ok', imu_fallback: 'bad',
};

async function poll() {
  const r = await fetch('/status.json');
  const s = await r.json();

  const badge = document.querySelector('#stageBadge');
  const stageInfo = {
    DRY_RUN: ['#a60', 'DRY RUN — 실제 명령 전송 안 함 (관찰 중)'],
    ARMED:   ['#960', 'ARMED — GO LIVE 대기 중 (아직 실제 명령 전송 안 함)'],
    LIVE:    ['#600', '⚠ LIVE — 실제 로봇으로 명령 전송 중'],
  };
  const [bg, label] = stageInfo[s.control_stage] || stageInfo.DRY_RUN;
  badge.innerHTML = `<span style="background:${bg}; color:#fff; padding:2px 8px; border-radius:4px; font-size:14px;">${label}</span>`;

  document.querySelector('#btnConfirm').disabled = (s.control_stage !== 'DRY_RUN');
  document.querySelector('#btnArm').disabled = (s.control_stage !== 'DRY_RUN' || s.route_aligned_heading_deg === null);
  document.querySelector('#btnGoLive').disabled = (s.control_stage !== 'ARMED');
  document.querySelector('#btnAbort').disabled = (s.control_stage === 'DRY_RUN');

  const gpsClass = s.gps_ok ? 'ok' : 'bad';
  const ageClass = (s.age_sec !== null && s.age_sec < 2.0) ? 'ok' : 'bad';
  const fixClass = (s.fix_quality !== null && s.fix_quality >= 2) ? 'ok' : 'bad';
  const gpsAgeClass = (s.gps_age_sec !== null && s.gps_age_sec < 1.5) ? 'ok' : 'bad';
  const routeDiffClass = (s.heading_route_diff_deg !== null && Math.abs(s.heading_route_diff_deg) < 30) ? 'ok' : 'bad';
  const latClass = (s.control_latency_ms !== null && s.control_latency_ms < 500) ? 'ok' : 'bad';
  const srcLabel = SOURCE_LABEL[s.map_heading_source] || (s.map_heading_source || '—');
  const srcClass = SOURCE_CLASS[s.map_heading_source] || '';
  const readyClass = (s.gps_accumulated_path_m !== null && s.gps_accumulated_path_m >= 1.5) ? 'ok' : 'warn';
  const osrmClass = (s.osrm_fallback === false) ? 'ok' : (s.osrm_fallback === true ? 'bad' : '');

  document.querySelector('#statusLoc tbody').innerHTML = `
    <tr><td>GPS</td><td class="${gpsClass}">${s.lat}, ${s.lon} ${s.gps_ok ? '' : '(FIX 없음!)'}</td></tr>
    <tr><td>fix_quality</td><td class="${fixClass}">${s.fix_quality === null ? '—' : s.fix_quality} <span style="color:#888">(NMEA GGA: 0=無, 1=SPS, 2=DGPS)</span></td></tr>
    <tr><td>GPS 데이터 나이</td><td class="${gpsAgeClass}">${fmtNum(s.gps_age_sec, 2)}s</td></tr>
    <tr><td>IMU heading (raw)</td><td>${fmtNum(s.orientation_deg_raw, 1)}&deg;</td></tr>
    <tr><td>GPS 궤적 heading</td><td>${s.gps_heading_deg === null ? '—' : fmtNum(s.gps_heading_deg, 1) + '&deg;'}</td></tr>
    <tr><td><b>heading source</b></td><td class="${srcClass}"><b>${srcLabel}</b></td></tr>
    <tr><td>route-aligned 확정값</td><td>${s.route_aligned_heading_deg === null ? '(미확정)' : fmtNum(s.route_aligned_heading_deg, 1) + '&deg;'}</td></tr>
    <tr><td><b>최종 사용 heading</b></td><td><b>${fmtNum(s.heading_deg, 1)}&deg;</b></td></tr>
    <tr><td>GPS heading 준비도</td><td class="${readyClass}">누적 ${fmtNum(s.gps_accumulated_path_m, 2)}m / 순변위 ${fmtNum(s.gps_net_displacement_m, 2)}m (기준 1.5m/0.3m)</td></tr>
  `;
  document.querySelector('#statusRoute tbody').innerHTML = `
    <tr><td>OSRM</td><td class="${osrmClass}">${s.osrm_fallback === null ? '(미초기화)' : (s.osrm_fallback ? 'FALLBACK(직선 경로)' : '정상')}</td></tr>
    <tr><td>snap 거리(출발/목표)</td><td>${fmtNum(s.start_snap_m, 2)}m / ${fmtNum(s.goal_snap_m, 2)}m</td></tr>
    <tr><td>route bearing (5m, debug)</td><td>${s.route_bearing_deg === null ? '—' : fmtNum(s.route_bearing_deg, 1) + '&deg;'}</td></tr>
    <tr><td>heading - route 차이</td><td class="${routeDiffClass}">${s.heading_route_diff_deg === null ? '—' : fmtNum(s.heading_route_diff_deg, 1) + '&deg;'}</td></tr>
    <tr><td>지도 회전각</td><td>${s.map_rotation_deg === null ? '—' : fmtNum(s.map_rotation_deg, 1) + '&deg;'}</td></tr>
  `;
  document.querySelector('#statusModel tbody').innerHTML = `
    <tr><td>target waypoint (x,y)</td><td>${s.target_waypoint_xy ? `(${fmtNum(s.target_waypoint_xy[0],3)}, ${fmtNum(s.target_waypoint_xy[1],3)}) m` : '—'}</td></tr>
    <tr><td>계산된 명령</td><td>linear=${s.computed_linear} m/s, angular=${s.computed_angular} rad/s</td></tr>
    <tr><td>실제 전송된 명령</td><td>linear=${s.linear} m/s, angular=${s.angular} rad/s${s.control_stage !== 'LIVE' ? ' <span class="warn">(전송 안 함 — ' + s.control_stage + ')</span>' : ''}</td></tr>
    <tr><td>/control 왕복시간</td><td class="${latClass}">${fmtNum(s.control_latency_ms, 0)} ms</td></tr>
    <tr><td>루프 주기</td><td>${s.loop_hz} Hz</td></tr>
    <tr><td>tick_id</td><td>${s.tick_id === null ? '—' : s.tick_id}</td></tr>
    <tr><td>마지막 갱신</td><td class="${ageClass}">${s.age_sec}s 전</td></tr>
    <tr><td>누적 에러</td><td class="${s.n_errors > 0 ? 'bad' : 'ok'}">${s.n_errors}</td></tr>
  `;
  document.querySelector('#logs').textContent = s.logs.join('\\n');
}
poll(); setInterval(poll, 1000);
</script>
</body></html>
"""


# 2026-09-25 추가: 대시보드 버튼 → 제어 루프 스레드로 명령을 전달하는 경로.
# HTTP 서버는 별도 스레드에서 돌므로, 제어 루프가 쓰는 복잡한 객체(map_builder,
# route 캐시 등)를 HTTP 핸들러가 직접 건드리지 않게 하려고 단순 문자열 명령만
# 큐에 넣는다 — 실제 상태 변경은 전부 제어 루프 스레드(step() 앞단)에서
# _apply_pending_commands()가 큐를 비우며 순차 처리(단일 소비자라 락 불필요).
ALLOWED_COMMANDS = {"confirm_alignment", "arm", "go_live", "abort"}


class _Handler(BaseHTTPRequestHandler):
    state: DeploymentState = None      # set by start_debug_server
    cmd_queue: "queue.Queue" = None    # set by start_debug_server

    def log_message(self, fmt, *args):
        pass  # 콘솔에 access log 안 찍히게

    def do_GET(self):
        if self.path.startswith("/status.json"):
            body = json.dumps(self.state.snapshot_status()).encode()
            self._send(body, "application/json")
        elif self.path.startswith("/frame.jpg"):
            self._send(self.state.snapshot_camera_jpeg(), "image/jpeg")
        elif self.path.startswith("/map.jpg"):
            self._send(self.state.snapshot_map_jpeg(), "image/jpeg")
        elif self.path.startswith("/map_northup.jpg"):
            self._send(self.state.snapshot_map_northup_jpeg(), "image/jpeg")
        elif self.path.startswith("/traj.jpg"):
            self._send(self.state.snapshot_traj_jpeg(), "image/jpeg")
        else:
            body = _PAGE_HTML.replace("__TS__", str(int(time.time() * 1000))).encode()
            self._send(body, "text/html; charset=utf-8")

    def do_POST(self):
        cmd = self.path.strip("/").split("?")[0]
        if cmd in ALLOWED_COMMANDS and self.cmd_queue is not None:
            self.cmd_queue.put(cmd)
            self.state.log(f"[dashboard] 명령 접수: {cmd} (제어 루프에서 다음 tick에 처리)")
            self._send(json.dumps({"ok": True, "queued": cmd}).encode(), "application/json")
        else:
            self._send(json.dumps({"ok": False, "error": "unknown command"}).encode(),
                        "application/json", status=400)

    def _send(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def start_debug_server(state: DeploymentState, cmd_queue: "queue.Queue" = None,
                        port: int = 8080) -> ThreadingHTTPServer:
    """백그라운드 스레드에서 대시보드 서버 시작. 서버 인스턴스 반환(필요 시 .shutdown()).
    cmd_queue: ARM/GO LIVE/정렬 확인/ABORT 버튼이 넣는 명령 큐 (제어 루프가 소비).
    None으로 두면(예: 순수 뷰어 용도) 버튼 클릭이 전부 무시됨."""
    handler_cls = type("BoundHandler", (_Handler,), {"state": state, "cmd_queue": cmd_queue})
    server = ThreadingHTTPServer(("0.0.0.0", port), handler_cls)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[debug_web] 대시보드: http://0.0.0.0:{port}  (같은 네트워크의 다른 기기에서는 "
          f"이 머신의 실제 IP로 접속)")
    return server
