"""
generate_debug_panel_capture.py

캡스톤 발표용 "주행 전 Debug Panel 검증 화면" 캡처.

실제 로봇/SDK 연결 없이, 2026-09-25 오프라인 스모크 테스트로 이미 저장된 실제
데이터(deployment/logs/deploy_20260925_164947.jsonl, tick_id=11)를 그대로
debug_web.py의 DeploymentState에 채워 넣고, 실제 debug_web.py 서버를 띄운 뒤
헤드리스 브라우저(pyppeteer, earth-rovers-sdk와 동일 라이브러리)로 실제 렌더링된
페이지를 스크린샷한다 — 즉 "이 그림이 곧 실제 대시보드 화면"이다.

사용한 데이터 (전부 실측/실제 코드 실행 결과 — 임의로 만든 값 없음):
  - run_id: 20260925_164947 (offline_smoke_test.py의 dry_run=False 실행)
  - tick_id: 11
  - 카메라: 그 tick의 context_frame_paths 중 최신 프레임 (2026-09-18 실제 배포
    스크린캐스트에서 추출한 진짜 카메라 프레임)
  - 지도: map_replay_path로 저장된 실제 model-input map + 그 tick의 실제
    pred_xy_m으로 다시 그린 예측 궤적 오버레이(LiveMapBuilder.draw_predicted_trajectory,
    저장 당시와 동일한 함수)
  - heading/route/control 값: 전부 그 tick의 JSONL 레코드 값 그대로
  - loop_hz / /control latency: 이 스모크 테스트는 requests.get/post를 mock했기
    때문에(진짜 네트워크 없음) 값 자체는 비현실적으로 낮음(0.1ms) — 그대로 화면에
    보여주되, README에 "이 값은 mock 환경값이며, 2026-09-25 실제 로봇 연결 상태에서
    직접 측정한 /control 왕복시간은 289~415ms였다"라고 명시함(수치 조작 없음)

실행: (frodobot conda env)
    python3 docs/deployment_visualization/02_debug_panel/generate_debug_panel_capture.py
출력: docs/deployment_visualization/02_debug_panel/debug_panel_capture.png
"""
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "deployment"))
sys.path.insert(0, str(REPO_ROOT / "osm_pipeline" / "py"))

from PIL import Image
from debug_web import DeploymentState, start_debug_server
from build_live_map import LiveMapBuilder

RUN_ID = "20260925_165933"  # dry_run=True run — pre-drive 체크 시나리오를 보여주기 위해 선택
TICK_ID = 11
JSONL_PATH = REPO_ROOT / "deployment" / "logs" / f"deploy_{RUN_ID}.jsonl"
LOG_DIR = REPO_ROOT / "deployment" / "logs"
OUT_DIR = Path(__file__).resolve().parent


def load_tick():
    lines = [json.loads(l) for l in open(JSONL_PATH)]
    step = next(l for l in lines if l.get("type") == "step" and l.get("tick_id") == TICK_ID)
    ctrl = next(l for l in lines if l.get("type") == "control_sent"
                and abs(l["ts"] - step["ts"]) < 1.0)
    steps = [l for l in lines if l.get("type") == "step"]
    idx = steps.index(step)
    loop_dt = step["ts"] - steps[idx - 1]["ts"] if idx > 0 else None
    return step, ctrl, loop_dt


def build_state(step, ctrl, loop_dt):
    state = DeploymentState()

    cam_path = LOG_DIR / step["context_frame_paths"][-1]
    camera_img = Image.open(cam_path).convert("RGB")

    map_path = LOG_DIR / step["map_replay_path"]
    map_np = np.array(Image.open(map_path).convert("RGB"))
    builder = LiveMapBuilder(map_range_m=20.0)
    pred_xy_m = np.array(step["pred_xy_m"])
    map_with_pred = builder.draw_predicted_trajectory(map_np, pred_xy_m)

    state.update(
        camera_img=camera_img,
        map_img=map_with_pred,
        lat=step["lat"], lon=step["lon"],
        heading_deg=step["smoothed_heading_deg"],
        orientation_deg_raw=step["imu_heading_deg"],
        linear=ctrl["linear"], angular=ctrl["angular"],
        computed_linear=ctrl["computed_linear"], computed_angular=ctrl["computed_angular"],
        pred_xy_m=pred_xy_m,
        loop_hz=round(1.0 / loop_dt, 2) if loop_dt else None,
        fix_quality=step["frodobot_raw"].get("fix_quality"),
        gps_data_ts=step["frodobot_raw"].get("timestamp"),
        gps_heading_deg=step["gps_heading_deg"],
        route_bearing_deg=step["route_bearing_deg"],
        heading_route_diff_deg=step["heading_route_diff_deg"],
        map_rotation_deg=step["map_rotation_deg"],
        target_waypoint_xy=tuple(step["target_waypoint_xy"]),
        control_latency_ms=ctrl["latency_ms"],
        tick_id=step["tick_id"],
        dry_run=ctrl["dry_run"],
    )
    return state


async def capture(port=8099):
    from pyppeteer import launch
    browser = await launch(headless=True, args=["--no-sandbox"],
                            executablePath="/usr/bin/google-chrome")
    page = await browser.newPage()
    await page.setViewport({"width": 1500, "height": 1500})
    await page.goto(f"http://127.0.0.1:{port}/", waitUntil="networkidle0")
    await asyncio.sleep(1.2)  # poll()이 status.json을 한 번 더 받아오도록 대기
    out_path = OUT_DIR / "debug_panel_capture.png"
    await page.screenshot({"path": str(out_path), "fullPage": True})
    await browser.close()
    return out_path


def main():
    step, ctrl, loop_dt = load_tick()
    state = build_state(step, ctrl, loop_dt)
    port = 8099
    server = start_debug_server(state, port=port)
    try:
        out_path = asyncio.get_event_loop().run_until_complete(capture(port))
        print(f"saved: {out_path}")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
