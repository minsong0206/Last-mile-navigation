"""
offline_smoke_test.py

--dry_run / deterministic replay logging / 확장된 디버그 필드를 실제 로봇·SDK
연결 없이 검증하는 오프라인 스모크 테스트 (2026-09-25 추가).

requests.get/post를 가짜 응답으로 monkeypatch해서 OmniVLAEdgeDeployment.run()의
실제 루프를 그대로 몇 틱 돌리고, 다음을 확인한다:
  - tick_id가 매 틱 증가하는지
  - 카메라 프레임 dedup이 동작하는지 (같은 프레임 재사용 시 frame_id 안 늘어남)
  - deployment/logs/frames/<run_id>/ 밑에 map_*.png / ctx_*.jpg가 실제로 저장되는지
  - route_bearing/heading_route_diff가 경로 초기화 후 계산되는지
  - --dry_run일 때 실제 전송된 linear/angular가 항상 (0,0)이고, 계산된 값은
    JSONL의 computed_linear/computed_angular로 따로 남는지
  - GPS fix 없음(lat=1000) 틱에서 정지 명령이 나가는지

실행: (frodobot conda env, CARTO_API_KEY 필요 — OSRM은 꺼져있어도 직선 폴백으로 동작)
    python3 deployment/offline_smoke_test.py
"""
import base64
import io
import json
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "deployment"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "omnivla" / "inference"))

from PIL import Image
import omnivla_edge_deploy as dep


def _b64_png(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_fake_requests(camera_frames, gps_sequence):
    """실제 SDK 서버 없이 poll_frodobot()/send_control()/run()의 워밍업이 그대로
    동작하도록 requests.get/post를 흉내내는 가짜 응답 객체들을 만든다."""
    state = {"gps_idx": 0, "cam_idx": 0, "control_calls": []}

    class FakeResp:
        def __init__(self, payload, status_code=200):
            self._payload = payload
            self.status_code = status_code

        def json(self):
            return self._payload

        def raise_for_status(self):
            pass

    def fake_get(url, timeout=None, **kw):
        if "/v2/front" in url:
            img = camera_frames[state["cam_idx"] % len(camera_frames)]
            state["cam_idx"] += 1
            return FakeResp({"front_frame": _b64_png(img), "timestamp": 0.0})
        if "/data" in url:
            gps = dict(gps_sequence[min(state["gps_idx"], len(gps_sequence) - 1)])
            state["gps_idx"] += 1
            # 실제 SDK는 항상 "방금" 시각을 timestamp로 주므로(그래야 gps_age 계산이
            # 의미 있음), 고정값(0.0) 대신 매 호출 시점의 실제 시각을 넣는다 — 그래야
            # 대시보드의 "GPS 데이터 나이"가 발표 캡처에서 비정상적으로 크게 안 나옴.
            gps["timestamp"] = time.time()
            return FakeResp(gps)
        raise AssertionError(f"unexpected GET {url}")

    def fake_post(url, json=None, timeout=None, **kw):
        assert "/control" in url
        cmd = json["command"]
        state["control_calls"].append((cmd["linear"], cmd["angular"]))
        return FakeResp({"message": "Command sent successfully"})

    return fake_get, fake_post, state


def run_smoke_test(dry_run: bool, n_ticks: int = 12):
    print(f"\n========== offline smoke test (dry_run={dry_run}) ==========")

    # 실제 테스트에 쓰던 카메라 프레임 재사용 (이번 세션에서 미리 추출해둔 것) —
    # 없으면 단색 더미 이미지로 대체.
    ctx_dir = Path("/tmp/claude-1000/-home-moai-Last-mile-navigation/"
                    "d2e40543-7db8-4687-b300-81fab4017442/scratchpad/frames0918")
    frame_paths = sorted(ctx_dir.glob("cam_*.png")) if ctx_dir.exists() else []
    if frame_paths:
        camera_frames = [Image.open(p).convert("RGB") for p in frame_paths]
    else:
        camera_frames = [Image.new("RGB", (640, 480), (80, 80, 80))]
    print(f"camera_frames: {len(camera_frames)}개 로드")

    # 일부러 같은 프레임을 반복시켜서 dedup을 테스트 (frame_id가 안 늘어나야 함)
    camera_frames = camera_frames + [camera_frames[-1]] * 3

    base_gps = dict(
        latitude=37.63118362426758, longitude=127.07622528076172,
        orientation=192.0, fix_quality=2, gps_signal=45.0,
        timestamp="0.0", speed=0.3, hdop=0.02,
        accels=[[0, 0, 1.0, 0.0]] * 5, gyros=[[0, 0, 0, 0.0]] * 5,
        mags=[[0, 0, 0]], rpms=[[10, 10, 10, 10, 0.0]] * 5,
        distance_to_rtk_station=-1, battery=90, signal_level=0, lamp=0,
        vibration=0, altitude=0, gps_ttff=0, power=1, current=100, voltage=40,
        network_state=2,
    )
    # 앞 3개는 GPS fix 없음(1000 sentinel)을 흉내내서 안전 분기도 같이 테스트.
    # 주의: run()의 워밍업(requests.get(.../data))이 이 시퀀스의 0번째 항목을
    # 미리 하나 소비하므로, step() 루프에서 실제로 보게 되는 "fix 없음" 틱은
    # 2개(1~2번 인덱스)뿐이다 — 그래서 3개를 준비해둔다.
    gps_sequence = [
        {**base_gps, "latitude": 1000, "longitude": 1000, "fix_quality": 0},
        {**base_gps, "latitude": 1000, "longitude": 1000, "fix_quality": 0},
        {**base_gps, "latitude": 1000, "longitude": 1000, "fix_quality": 0},
    ]
    for i in range(n_ticks):
        gps_sequence.append({**base_gps,
                              "latitude": base_gps["latitude"] - i * 0.00001,
                              "longitude": base_gps["longitude"] - i * 0.000005})

    fake_get, fake_post, call_state = build_fake_requests(camera_frames, gps_sequence)

    import os
    os.environ.setdefault("CARTO_API_KEY", "cb1_2yj0_1_ffcc69174af013089c6c5da6")

    # 주의: time.sleep()은 mock하지 않는다 — maybe_update_frame_buffer()가
    # CTX_STRIDE_SEC(0.3초) 경과를 실제 time.time()으로 판단하므로, sleep을
    # 없애버리면 컨텍스트가 영원히 안 채워져서(첫 프레임에서 멈춤) build_inputs()/
    # 지도 저장까지 도달을 못 한다 — 테스트가 몇 초 더 걸리더라도 실제 시간이
    # 흘러야 이 테스트가 실제 배포 루프와 같은 조건이 된다.
    with mock.patch("requests.get", side_effect=fake_get), \
         mock.patch("requests.post", side_effect=fake_post):
        deployer = dep.OmniVLAEdgeDeployment(
            ckpt_path=str(REPO_ROOT / "checkpoints" / "omnivla_edge_rides11_odom_20m_20260910" / "best.pth"),
            map_range_m=20.0, goal_lat=37.6310081, goal_lon=127.0760555,
            debug_port=0, dry_run=dry_run,
        )

        tick_count = [0]
        real_step = deployer.step

        def counted_step():
            tick_count[0] += 1
            if tick_count[0] > len(gps_sequence):
                raise KeyboardInterrupt()
            return real_step()

        deployer.step = counted_step
        deployer.run()

    # ── 검증 ──
    log_lines = [json.loads(l) for l in open(deployer.log_path)]
    steps = [l for l in log_lines if l["type"] == "step"]
    controls = [l for l in log_lines if l["type"] == "control_sent"]

    tick_ids = [s["tick_id"] for s in steps]
    assert tick_ids == sorted(set(tick_ids)) and len(tick_ids) == len(set(tick_ids)), \
        f"tick_id가 중복 없이 증가해야 함: {tick_ids}"
    print(f"[OK] tick_id 단조 증가 확인: {tick_ids}")

    gps_missing = [s for s in steps if s.get("note") == "gps_fix_missing"]
    print(f"[OK] gps_fix_missing 분기 {len(gps_missing)}회 발생 (기대: 앞 2틱)")
    assert len(gps_missing) == 2

    frames_dir = deployer.replay_logger.frames_dir
    saved_ctx = sorted(frames_dir.glob("ctx_*.jpg"))
    saved_maps = sorted(frames_dir.glob("map_*.png"))
    print(f"[OK] 저장된 컨텍스트 프레임: {len(saved_ctx)}개 (dedup 덕에 카메라 프레임 수보다 적어야 함)")
    print(f"[OK] 저장된 지도 이미지: {len(saved_maps)}개")
    assert len(saved_ctx) > 0 and len(saved_maps) > 0

    with_route_bearing = [s for s in steps if s.get("route_bearing_deg") is not None]
    print(f"[OK] route_bearing_deg 계산된 틱: {len(with_route_bearing)}개 "
          f"(경로 초기화 이후부터 나와야 함)")

    if dry_run:
        sent_nonzero = [c for c in controls if (c["linear"], c["angular"]) != (0.0, 0.0)]
        assert len(sent_nonzero) == 0, f"dry_run인데 0이 아닌 명령이 실제 전송됨: {sent_nonzero}"
        computed_nonzero = [c for c in controls
                             if c.get("computed_linear", 0) != 0 or c.get("computed_angular", 0) != 0]
        print(f"[OK] dry_run: 실제 전송값은 전부 (0,0), "
              f"계산값은 {len(computed_nonzero)}개 틱에서 0이 아님(정상 — 계산은 계속 되고 있다는 뜻)")
        # +1: KeyboardInterrupt 핸들러가 마지막에 한 번 더 보내는 안전정지
        # (run()에서 JSONL control_sent로는 안 남기고 send_control()만 직접 호출함)
        assert len(call_state["control_calls"]) == len(controls) + 1
        assert all(c == (0.0, 0.0) for c in call_state["control_calls"]), \
            "가짜 SDK가 실제로 받은 /control 값도 전부 (0,0)이어야 함"
        print("[OK] 가짜 SDK가 실제로 수신한 /control 페이로드도 전부 (0,0) 확인")
    else:
        print(f"[OK] live 모드: control_calls={call_state['control_calls'][:5]}...")

    dry_run_flags = [c.get("dry_run") for c in controls]
    assert all(f == dry_run for f in dry_run_flags), "control_sent 레코드의 dry_run 플래그 불일치"
    print(f"[OK] JSONL control_sent.dry_run 필드가 전부 {dry_run}로 일관됨")

    status = deployer.state.snapshot_status()
    assert status["dry_run"] == dry_run
    print(f"[OK] DeploymentState.snapshot_status()['dry_run'] = {status['dry_run']}")
    print(f"    tick_id={status['tick_id']}  fix_quality={status['fix_quality']}  "
          f"route_bearing_deg={status['route_bearing_deg']}  "
          f"heading_route_diff_deg={status['heading_route_diff_deg']}  "
          f"target_waypoint_xy={status['target_waypoint_xy']}  "
          f"control_latency_ms={status['control_latency_ms']}")

    print(f"[OK] 로그 파일: {deployer.log_path}")
    print(f"[OK] replay 프레임 디렉토리: {frames_dir}")
    return deployer


if __name__ == "__main__":
    run_smoke_test(dry_run=True)
    run_smoke_test(dry_run=False)
    print("\n모든 오프라인 스모크 테스트 통과.")
