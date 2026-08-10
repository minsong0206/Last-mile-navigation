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
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image


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
        self.linear = 0.0
        self.angular = 0.0
        self.pred_xy_m = None       # (8,2) numpy, 예측 웨이포인트
        self.last_update_ts = None
        self.loop_hz = None
        self._logs = deque(maxlen=log_maxlen)
        self._errors = deque(maxlen=log_maxlen)

    def update(self, camera_img=None, map_img=None, lat=None, lon=None,
               heading_deg=None, linear=None, angular=None, pred_xy_m=None,
               loop_hz=None):
        with self._lock:
            if camera_img is not None: self.camera_img = camera_img
            if map_img is not None: self.map_img = map_img
            if lat is not None: self.lat = lat
            if lon is not None: self.lon = lon
            if heading_deg is not None: self.heading_deg = heading_deg
            if linear is not None: self.linear = linear
            if angular is not None: self.angular = angular
            if pred_xy_m is not None: self.pred_xy_m = pred_xy_m
            if loop_hz is not None: self.loop_hz = loop_hz
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
            return {
                "lat": self.lat, "lon": self.lon,
                "heading_deg": self.heading_deg,
                "linear": self.linear, "angular": self.angular,
                "loop_hz": self.loop_hz,
                "age_sec": round(age, 2) if age is not None else None,
                "pred_xy_m": self.pred_xy_m.tolist() if self.pred_xy_m is not None else None,
                "logs": list(self._logs)[-30:],
                "n_errors": len(self._errors),
                "gps_ok": (self.lat is not None and self.lat != 1000 and self.lon != 1000),
            }

    def snapshot_camera_jpeg(self):
        with self._lock:
            img = self.camera_img
        return _to_jpeg(img)

    def snapshot_map_jpeg(self):
        with self._lock:
            img = self.map_img
        return _to_jpeg(img)


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
<meta http-equiv="refresh" content="1">
<style>
body { font-family: monospace; background:#111; color:#eee; margin:20px; }
img { width: 320px; height: 320px; object-fit: contain; background:#000; border:1px solid #444; }
.row { display:flex; gap:20px; flex-wrap:wrap; }
.card { background:#1c1c1c; padding:12px; border-radius:8px; }
table { border-collapse: collapse; }
td { padding: 2px 10px 2px 0; }
.ok { color:#6f6; } .bad { color:#f66; }
pre { background:#000; padding:8px; height:300px; overflow-y:auto; font-size:12px; }
</style></head>
<body>
<h2>OmniVLA-Edge 실시간 배포 모니터</h2>
<div class="row">
  <div class="card"><b>카메라</b><br><img src="/frame.jpg?t=__TS__"></div>
  <div class="card"><b>실시간 생성 지도</b><br><img src="/map.jpg?t=__TS__"></div>
  <div class="card">
    <b>상태</b>
    <table id="status"><tbody></tbody></table>
  </div>
</div>
<div class="card" style="margin-top:16px;">
  <b>로그 (최근 30줄)</b>
  <pre id="logs"></pre>
</div>
<script>
async function poll() {
  const r = await fetch('/status.json');
  const s = await r.json();
  const tb = document.querySelector('#status tbody');
  const gpsClass = s.gps_ok ? 'ok' : 'bad';
  const ageClass = (s.age_sec !== null && s.age_sec < 2.0) ? 'ok' : 'bad';
  tb.innerHTML = `
    <tr><td>GPS</td><td class="${gpsClass}">${s.lat}, ${s.lon} ${s.gps_ok ? '' : '(FIX 없음!)'}</td></tr>
    <tr><td>heading</td><td>${s.heading_deg}&deg;</td></tr>
    <tr><td>제어명령</td><td>linear=${s.linear} m/s, angular=${s.angular} rad/s</td></tr>
    <tr><td>루프 주기</td><td>${s.loop_hz} Hz</td></tr>
    <tr><td>마지막 갱신</td><td class="${ageClass}">${s.age_sec}s 전</td></tr>
    <tr><td>누적 에러</td><td class="${s.n_errors > 0 ? 'bad' : 'ok'}">${s.n_errors}</td></tr>
  `;
  document.querySelector('#logs').textContent = s.logs.join('\\n');
}
poll(); setInterval(poll, 1000);
</script>
</body></html>
"""


class _Handler(BaseHTTPRequestHandler):
    state: DeploymentState = None  # set by start_debug_server

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
        else:
            body = _PAGE_HTML.replace("__TS__", str(int(time.time() * 1000))).encode()
            self._send(body, "text/html; charset=utf-8")

    def _send(self, body, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def start_debug_server(state: DeploymentState, port: int = 8080) -> ThreadingHTTPServer:
    """백그라운드 스레드에서 대시보드 서버 시작. 서버 인스턴스 반환(필요 시 .shutdown())."""
    handler_cls = type("BoundHandler", (_Handler,), {"state": state})
    server = ThreadingHTTPServer(("0.0.0.0", port), handler_cls)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[debug_web] 대시보드: http://0.0.0.0:{port}  (같은 네트워크의 다른 기기에서는 "
          f"이 머신의 실제 IP로 접속)")
    return server
