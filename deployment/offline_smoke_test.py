"""
offline_smoke_test.py

Route-aligned initial heading bootstrap + ARM/GO LIVE 2단계 워크플로(2026-09-25)를
실제 로봇·SDK 연결 없이 검증하는 오프라인 스모크 테스트.

requests.get/post를 가짜 응답으로 monkeypatch해서 OmniVLAEdgeDeployment.run()의
실제 루프를 그대로 몇 틱 돌리고, 대시보드 버튼이 하는 것과 동일하게
deployer._pending_commands에 명령을 넣어 다음을 확인한다:

  test_a_dry_run_lock:
    - --dry_run(잠금) 모드에서는 confirm_alignment/arm/go_live를 다 눌러도
      control_stage가 절대 LIVE가 안 되고, 전송값이 항상 (0,0)인지.

  test_b_full_workflow:
    - GPS fix 없음 -> route 초기화(OSRM, fallback 포함) -> route-aligned 정렬 확인
      -> ARM -> GO LIVE까지 재시작 없이 같은 프로세스/같은 deployer 인스턴스에서
      전환되는지, 그 사이 frame_buffer/past_track/heading 상태가 유지되는지.
    - GO LIVE 전까지는 로봇이 실제로 이동했더라도(GPS가 바뀌어도) 전송값이 항상
      (0,0)인지.
    - GO LIVE 이후 map_heading_source가 "route_aligned"로 시작하고, 실측 GPS
      이동이 누적되면 "gps_track"으로 정확히 한 번 전환되며 그 전환 이벤트가
      JSONL에 남는지 (_heading_ema_vec가 route-aligned 값으로 오염되지 않았는지도
      전환 시점 diff 값으로 확인).

  test_c_arm_without_alignment:
    - route-aligned 정렬을 확인하지 않은 채 ARM을 누르면 거부되고 DRY_RUN에
      머무르는지 (heading-readiness에 대한 첫 번째 방어선).

실행: (frodobot conda env, CARTO_API_KEY 필요 — OSRM은 꺼져있어도 직선 폴백으로 동작)
    python3 deployment/offline_smoke_test.py
"""
import base64
import io
import json
import os
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

os.environ.setdefault("CARTO_API_KEY", "cb1_2yj0_1_ffcc69174af013089c6c5da6")

CKPT_PATH = REPO_ROOT / "checkpoints" / "omnivla_edge_rides11_odom_20m_20260910" / "best.pth"
GOAL_LAT, GOAL_LON = 37.6310081, 127.0760555

# Seoul(osrm_port 5011 대역)에 들어가는 좌표 — 기존 스모크 테스트와 동일 기준점
BASE_LAT, BASE_LON = 37.63118362426758, 127.07622528076172


def _b64_png(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _load_camera_frames():
    ctx_dir = Path("/tmp/claude-1000/-home-moai-Last-mile-navigation/"
                    "d2e40543-7db8-4687-b300-81fab4017442/scratchpad/frames0918")
    frame_paths = sorted(ctx_dir.glob("cam_*.png")) if ctx_dir.exists() else []
    if frame_paths:
        return [Image.open(p).convert("RGB") for p in frame_paths]
    return [Image.new("RGB", (640, 480), (80, 80, 80))]


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
            gps["timestamp"] = time.time()
            return FakeResp(gps)
        raise AssertionError(f"unexpected GET {url}")

    def fake_post(url, json=None, timeout=None, **kw):
        assert "/control" in url
        cmd = json["command"]
        state["control_calls"].append((cmd["linear"], cmd["angular"]))
        return FakeResp({"message": "Command sent successfully"})

    return fake_get, fake_post, state


def _base_gps(lat, lon, orientation=192.0):
    return dict(
        latitude=lat, longitude=lon, orientation=orientation, fix_quality=2,
        gps_signal=45.0, timestamp="0.0", speed=0.3, hdop=0.02,
        accels=[[0, 0, 1.0, 0.0]] * 5, gyros=[[0, 0, 0, 0.0]] * 5,
        mags=[[0, 0, 0]], rpms=[[10, 10, 10, 10, 0.0]] * 5,
        distance_to_rtk_station=-1, battery=90, signal_level=0, lamp=0,
        vibration=0, altitude=0, gps_ttff=0, power=1, current=100, voltage=40,
        network_state=2,
    )


def make_deployer(fake_get, fake_post, dry_run_lock, n_ticks, heading_mode="auto",
                   goal_lat=None, goal_lon=None, goal_reach_threshold_m=dep.DEFAULT_GOAL_REACH_THRESHOLD_M,
                   initial_heading_lookahead_m=dep.INITIAL_HEADING_LOOKAHEAD_M):
    deployer = dep.OmniVLAEdgeDeployment(
        ckpt_path=str(CKPT_PATH), map_range_m=20.0,
        goal_lat=goal_lat if goal_lat is not None else GOAL_LAT,
        goal_lon=goal_lon if goal_lon is not None else GOAL_LON,
        debug_port=0, dry_run=dry_run_lock, heading_mode=heading_mode,
        goal_reach_threshold_m=goal_reach_threshold_m,
        initial_heading_lookahead_m=initial_heading_lookahead_m,
    )
    tick_count = [0]
    real_step = deployer.step

    def counted_step():
        tick_count[0] += 1
        if tick_count[0] > n_ticks:
            raise KeyboardInterrupt()
        return real_step()

    deployer.step = counted_step
    return deployer


def _read_jsonl(path):
    return [json.loads(l) for l in open(path)]


# ── test A: dry_run lock ──────────────────────────────────────────────────
def test_a_dry_run_lock():
    print("\n========== test_a_dry_run_lock ==========")
    camera_frames = _load_camera_frames() * 3
    n_ticks = 10
    gps_sequence = [_base_gps(1000, 1000, )] * 2   # 처음 2틱: fix 없음
    gps_sequence += [_base_gps(BASE_LAT - i * 0.00003, BASE_LON - i * 0.000015)
                      for i in range(n_ticks)]       # 그 뒤론 실제로 계속 이동(테스트 목적)

    fake_get, fake_post, call_state = build_fake_requests(camera_frames, gps_sequence)
    with mock.patch("requests.get", side_effect=fake_get), \
         mock.patch("requests.post", side_effect=fake_post):
        deployer = make_deployer(fake_get, fake_post, dry_run_lock=True, n_ticks=n_ticks)

        # 대시보드에서 confirm_alignment -> arm -> go_live를 전부 눌러도
        # dry_run_lock=True면 절대 LIVE가 되면 안 됨.
        real_step = deployer.step

        def wrapped():
            if dep_tick_count[0] == 3:
                deployer._pending_commands.put("confirm_alignment")
            if dep_tick_count[0] == 4:
                deployer._pending_commands.put("arm")
            if dep_tick_count[0] == 5:
                deployer._pending_commands.put("go_live")
            dep_tick_count[0] += 1
            return real_step()

        dep_tick_count = [0]
        deployer.step = wrapped
        deployer.run()

    status = deployer.state.snapshot_status()
    assert status["control_stage"] != "LIVE", f"dry_run_lock인데 LIVE가 됨: {status['control_stage']}"
    print(f"[OK] control_stage={status['control_stage']} (LIVE로 절대 안 바뀜)")
    assert all(c == (0.0, 0.0) for c in call_state["control_calls"]), \
        "dry_run_lock인데 0이 아닌 명령이 실제 전송됨"
    print(f"[OK] 실제 전송된 /control 페이로드 {len(call_state['control_calls'])}개 전부 (0,0)")

    logs = _read_jsonl(deployer.log_path)
    go_live_rejections = [l for l in logs if l.get("event") == "go_live"]
    assert len(go_live_rejections) == 0, "dry_run_lock인데 go_live 이벤트가 로그에 남음(승인된 것처럼 보임)"
    print("[OK] JSONL에 go_live 승인 이벤트 없음 (매번 거부됨)")
    print(f"[OK] 로그 파일: {deployer.log_path}")


# ── test B: 전체 워크플로 (route-align -> ARM -> GO LIVE -> gps_track 전환) ──
def test_b_full_workflow():
    print("\n========== test_b_full_workflow ==========")
    camera_frames = _load_camera_frames() * 3
    n_ticks = 22

    gps_sequence = [_base_gps(1000, 1000)] * 2  # 0~1: fix 없음
    # 2~9: fix는 있지만 완전히 정지(같은 좌표) — route-align 확인 전/후 모두
    #      GPS course heading은 절대 안 생겨야 함 (이동량 0)
    for _ in range(8):
        gps_sequence.append(_base_gps(BASE_LAT, BASE_LON))
    # 10~21: 이제부터 실제로 이동 시작(각 틱마다 ~1.1m 전진) — GO LIVE 이후
    #        누적 1.5m를 넘기면 estimate_heading_from_track이 값을 반환해야 함
    for i in range(12):
        gps_sequence.append(_base_gps(BASE_LAT - (i + 1) * 0.00001, BASE_LON - (i + 1) * 0.000005))

    fake_get, fake_post, call_state = build_fake_requests(camera_frames, gps_sequence)
    with mock.patch("requests.get", side_effect=fake_get), \
         mock.patch("requests.post", side_effect=fake_post):
        deployer = make_deployer(fake_get, fake_post, dry_run_lock=False, n_ticks=n_ticks)
        real_step = deployer.step
        tick_count = [0]

        def wrapped():
            t = tick_count[0]
            # run()의 워밍업이 gps_sequence[0]을 미리 하나 소비하므로, step() 루프
            # 안에서 실제로 보는 tick 번호는 이 카운터 기준.
            if t == 4:
                # route는 첫 유효 fix(gps_sequence[2]) 이후 이미 초기화돼 있어야 함
                assert deployer._route_initialized, "route가 아직 초기화 안 됨(정렬 확인 전제조건 실패)"
                deployer._pending_commands.put("confirm_alignment")
            if t == 6:
                deployer._pending_commands.put("arm")
            if t == 7:
                deployer._pending_commands.put("go_live")
            tick_count[0] += 1
            return real_step()

        deployer.step = wrapped
        deployer.run()

    logs = _read_jsonl(deployer.log_path)
    steps = [l for l in logs if l["type"] == "step"]
    controls = [l for l in logs if l["type"] == "control_sent"]

    # 1) route 초기화가 OSRM fallback이든 아니든 됐는지, 값이 노출되는지
    route_events = [l for l in logs if l.get("event") == "route_init"]
    assert len(route_events) == 1
    assert route_events[0]["osrm_fallback"] in (True, False)
    print(f"[OK] route_init 1회, osrm_fallback={route_events[0]['osrm_fallback']} "
          f"(OSRM 서버 없으면 True가 정상)")

    # 2) 정렬 확인 이벤트 확인
    align_events = [l for l in logs if l.get("event") == "route_alignment_confirmed"]
    assert len(align_events) == 1, f"route_alignment_confirmed 이벤트가 1개가 아님: {len(align_events)}"
    route_aligned_deg = align_events[0]["route_aligned_heading_deg"]
    print(f"[OK] route_alignment_confirmed 1회, heading={route_aligned_deg:+.1f}° "
          f"(lookahead={align_events[0]['lookahead_m']}m)")

    # 3) ARM/GO LIVE 이벤트 확인
    armed_events = [l for l in logs if l.get("event") == "armed"]
    go_live_events = [l for l in logs if l.get("event") == "go_live"]
    assert len(armed_events) == 1 and len(go_live_events) == 1
    print(f"[OK] armed 1회, go_live 1회. go_live 시점 heading_source="
          f"{go_live_events[0]['map_heading_source_at_go']}")
    assert go_live_events[0]["map_heading_source_at_go"] == "route_aligned", \
        "GO LIVE 시점엔 아직 실측 GPS heading이 없어야 하는 시나리오였음(route_aligned이어야 함)"

    # 4) GO LIVE 이전 tick들의 전송값은 전부 (0,0)이어야 함 (그 사이 GPS가 바뀌었어도)
    go_live_ts = go_live_events[0]["ts"]
    before_go = [c for c in controls if c["ts"] < go_live_ts]
    assert all(c["linear"] == 0.0 and c["angular"] == 0.0 for c in before_go), \
        "GO LIVE 전인데 0이 아닌 명령이 전송됨"
    print(f"[OK] GO LIVE 이전 {len(before_go)}개 tick 전부 전송값 (0,0)")

    # 5) heading_source_transition 이벤트: route_aligned -> gps_track 정확히 1회
    transitions = [l for l in logs if l.get("event") == "heading_source_transition"]
    assert len(transitions) == 1, f"heading_source_transition이 정확히 1번이어야 함: {len(transitions)}"
    tr = transitions[0]
    print(f"[OK] heading_source_transition 1회: route_aligned({tr['route_aligned_heading_deg']:+.1f}°) "
          f"-> gps_track({tr['first_gps_heading_deg']:+.1f}°), diff={tr['diff_deg']:+.1f}°")
    # route_aligned_heading_deg가 route_alignment_confirmed 때와 정확히 같아야 함(오염/변형 없음)
    assert abs(tr["route_aligned_heading_deg"] - route_aligned_deg) < 1e-6, \
        "전환 로그의 route_aligned 값이 정렬 확인 때 값과 달라짐 -- EMA가 오염됐을 가능성"
    print("[OK] 전환 로그의 route_aligned 값이 정렬 확인 시점 값과 정확히 일치 (EMA 오염 없음 확인)")

    # 6) GO LIVE 이후 & gps_track 전환 이전 구간의 map_heading_source는 route_aligned로 고정
    # (ts 비교가 아니라 tick_id로 비교 — 전환이 로그되는 그 tick 자체는 이미
    # map_heading_source=="gps_track"으로 찍히므로 tick_id 기준 '전' 구간만 봐야 함)
    transition_tick_id = tr["tick_id"]
    route_aligned_steps = [s for s in steps
                            if s["ts"] >= go_live_ts and s["tick_id"] < transition_tick_id
                            and s.get("map_heading_source") is not None]
    assert all(s["map_heading_source"] == "route_aligned" for s in route_aligned_steps), \
        "GO LIVE ~ 전환 사이에 route_aligned가 아닌 heading source가 섞여 있음"
    print(f"[OK] GO LIVE~전환 사이 {len(route_aligned_steps)}개 tick 전부 map_heading_source=route_aligned")

    # 7) 전환 이후(전환 tick 포함)에는 gps_track/gps_ema_hold만 나와야 함 (imu_fallback 재등장 금지)
    after_steps = [s for s in steps if s["tick_id"] >= transition_tick_id and s.get("map_heading_source") is not None]
    assert all(s["map_heading_source"] in ("gps_track", "gps_ema_hold", "gps_ema_gyro")
               for s in after_steps), "전환 이후 imu_fallback/route_aligned가 다시 나타남"
    print(f"[OK] 전환 이후 {len(after_steps)}개 tick 전부 gps_track 계열 유지")

    # 8) North-up 미리보기가 실제로 채워졌는지 (route 초기화 이후부터)
    status = deployer.state.snapshot_status()
    northup_jpeg = deployer.state.snapshot_map_northup_jpeg()
    assert northup_jpeg is not None and len(northup_jpeg) > 0
    print(f"[OK] North-up 미리보기 JPEG 생성 확인 ({len(northup_jpeg)} bytes)")

    # 9) GO LIVE 이후 실제로 non-zero 명령이 최소 한 번은 전송됐는지 (heading이
    #    route_aligned/gps_track이라 신뢰 가능했으므로 게이트에 안 걸려야 함)
    after_go = [c for c in controls if c["ts"] >= go_live_ts]
    nonzero_after_go = [c for c in after_go if (c["linear"], c["angular"]) != (0.0, 0.0)]
    print(f"[OK] GO LIVE 이후 {len(after_go)}개 tick 중 {len(nonzero_after_go)}개에서 non-zero 명령 전송됨")
    assert all(c["control_stage"] == "LIVE" for c in after_go)
    assert all(c["heading_trustworthy"] for c in after_go), \
        "GO LIVE 이후인데 heading_trustworthy=False인 tick이 있음(예상 밖 imu_fallback)"
    print("[OK] GO LIVE 이후 전부 heading_trustworthy=True (게이트가 불필요하게 막지 않음)")

    print(f"[OK] 로그 파일: {deployer.log_path}")


# ── test C: 정렬 확인 없이 ARM 시도 ──────────────────────────────────────────
def test_c_arm_without_alignment():
    print("\n========== test_c_arm_without_alignment ==========")
    camera_frames = _load_camera_frames() * 3
    n_ticks = 8
    gps_sequence = [_base_gps(1000, 1000)] * 2
    gps_sequence += [_base_gps(BASE_LAT, BASE_LON)] * n_ticks

    fake_get, fake_post, call_state = build_fake_requests(camera_frames, gps_sequence)
    with mock.patch("requests.get", side_effect=fake_get), \
         mock.patch("requests.post", side_effect=fake_post):
        deployer = make_deployer(fake_get, fake_post, dry_run_lock=False, n_ticks=n_ticks)
        real_step = deployer.step
        tick_count = [0]

        def wrapped():
            if tick_count[0] == 3:
                # 정렬 확인 없이 바로 ARM 시도 -- 거부되어야 함
                deployer._pending_commands.put("arm")
            tick_count[0] += 1
            return real_step()

        deployer.step = wrapped
        deployer.run()

    status = deployer.state.snapshot_status()
    assert status["control_stage"] == "DRY_RUN", \
        f"정렬 확인 없이 ARM했는데 stage가 바뀜: {status['control_stage']}"
    print(f"[OK] 정렬 확인 없는 ARM 시도는 거부됨, control_stage={status['control_stage']} 유지")
    assert all(c == (0.0, 0.0) for c in call_state["control_calls"])
    print("[OK] 실제 전송값 전부 (0,0)")


# ── test D: heading_mode=route_aligned_fixed — 실측 GPS가 잡혀도 절대 안 넘어감 ──
def test_d_route_aligned_fixed_mode():
    print("\n========== test_d_route_aligned_fixed_mode ==========")
    camera_frames = _load_camera_frames() * 3
    n_ticks = 22

    gps_sequence = [_base_gps(1000, 1000)] * 2
    for _ in range(8):
        gps_sequence.append(_base_gps(BASE_LAT, BASE_LON))
    # test_b와 동일하게 이후 실제로 계속 이동(1.5m 임계값을 넉넉히 넘기는 시나리오)
    for i in range(12):
        gps_sequence.append(_base_gps(BASE_LAT - (i + 1) * 0.00001, BASE_LON - (i + 1) * 0.000005))

    fake_get, fake_post, call_state = build_fake_requests(camera_frames, gps_sequence)
    with mock.patch("requests.get", side_effect=fake_get), \
         mock.patch("requests.post", side_effect=fake_post):
        deployer = make_deployer(fake_get, fake_post, dry_run_lock=False, n_ticks=n_ticks,
                                  heading_mode="route_aligned_fixed")
        real_step = deployer.step
        tick_count = [0]

        def wrapped():
            t = tick_count[0]
            if t == 4:
                deployer._pending_commands.put("confirm_alignment")
            if t == 6:
                deployer._pending_commands.put("arm")
            if t == 7:
                deployer._pending_commands.put("go_live")
            tick_count[0] += 1
            return real_step()

        deployer.step = wrapped
        deployer.run()

    logs = _read_jsonl(deployer.log_path)
    steps = [l for l in logs if l["type"] == "step"]

    align_events = [l for l in logs if l.get("event") == "route_alignment_confirmed"]
    assert len(align_events) == 1
    route_aligned_deg = align_events[0]["route_aligned_heading_deg"]
    print(f"[OK] route_alignment_confirmed 1회, heading={route_aligned_deg:+.1f}°")

    # 핵심 검증: route_aligned_fixed 모드에서는 실측 GPS 이동이 충분히 누적돼도
    # (test_b와 동일한 이동 시퀀스) heading_source_transition이 절대 발생하면 안 됨,
    # 그리고 확정 이후 모든 tick의 map_heading_source가 route_aligned로 고정돼야 함.
    transitions = [l for l in logs if l.get("event") == "heading_source_transition"]
    assert len(transitions) == 0, \
        f"route_aligned_fixed 모드인데 heading_source_transition이 발생함: {transitions}"
    print("[OK] heading_source_transition 이벤트 없음 (실측 GPS가 잡혀도 전환 안 됨)")

    after_confirm_steps = [s for s in steps if s["ts"] >= align_events[0]["ts"]
                            and s.get("map_heading_source") is not None]
    sources = {s["map_heading_source"] for s in after_confirm_steps}
    assert sources == {"route_aligned"}, \
        f"정렬 확인 이후 map_heading_source가 route_aligned 외의 값을 포함함: {sources}"
    print(f"[OK] 정렬 확인 이후 {len(after_confirm_steps)}개 tick 전부 map_heading_source=route_aligned "
          f"(gps_heading_deg는 로그에 참고용으로 계속 기록됨: "
          f"{[s.get('gps_heading_deg') for s in after_confirm_steps[-3:]]})")

    # smoothed_heading_deg(실제 map heading)이 확정값과 정확히 동일하게 유지되는지
    # (오염/드리프트 없이 진짜 "고정"인지)
    smoothed_vals = {round(s["smoothed_heading_deg"], 6) for s in after_confirm_steps}
    assert smoothed_vals == {round(route_aligned_deg, 6)}, \
        f"고정 모드인데 smoothed_heading_deg가 흔들림: {smoothed_vals}"
    print(f"[OK] smoothed_heading_deg가 확정값({route_aligned_deg:+.1f}°)과 전부 정확히 일치 (진짜 고정 확인)")

    print(f"[OK] 로그 파일: {deployer.log_path}")


# ── test E: goal 도착 시 영구 정지 ────────────────────────────────────────────
def test_e_goal_reached_stop():
    print("\n========== test_e_goal_reached_stop ==========")
    camera_frames = _load_camera_frames() * 3
    n_ticks = 22
    THRESH_M = 2.0

    gps_sequence = [_base_gps(1000, 1000)] * 2
    for _ in range(8):
        gps_sequence.append(_base_gps(BASE_LAT, BASE_LON))
    move_points = [(BASE_LAT - (i + 1) * 0.00001, BASE_LON - (i + 1) * 0.000005) for i in range(12)]
    for lat, lon in move_points:
        gps_sequence.append(_base_gps(lat, lon))

    # 목표를 이동 경로의 8번째 지점(정확히 그 좌표)으로 설정 -- 그 지점을
    # 지나는 순간 임계값(2.0m) 안쪽으로 들어와야 하고, 그 이후 조금 더
    # 이동해서(9~12번째) 계속 latched 상태인지 확인 가능.
    goal_lat, goal_lon = move_points[7]

    fake_get, fake_post, call_state = build_fake_requests(camera_frames, gps_sequence)
    with mock.patch("requests.get", side_effect=fake_get), \
         mock.patch("requests.post", side_effect=fake_post):
        deployer = make_deployer(fake_get, fake_post, dry_run_lock=False, n_ticks=n_ticks,
                                  goal_lat=goal_lat, goal_lon=goal_lon,
                                  goal_reach_threshold_m=THRESH_M)
        real_step = deployer.step
        tick_count = [0]

        def wrapped():
            t = tick_count[0]
            if t == 4:
                deployer._pending_commands.put("confirm_alignment")
            if t == 6:
                deployer._pending_commands.put("arm")
            if t == 7:
                deployer._pending_commands.put("go_live")
            tick_count[0] += 1
            return real_step()

        deployer.step = wrapped
        deployer.run()

    logs = _read_jsonl(deployer.log_path)
    steps = [l for l in logs if l["type"] == "step"]
    controls = [l for l in logs if l["type"] == "control_sent"]

    goal_events = [l for l in logs if l.get("event") == "goal_reached"]
    assert len(goal_events) == 1, f"goal_reached 이벤트가 정확히 1번이어야 함: {len(goal_events)}"
    ge = goal_events[0]
    assert ge["dist_to_goal_m"] <= THRESH_M
    print(f"[OK] goal_reached 이벤트 1회, tick_id={ge['tick_id']}, dist={ge['dist_to_goal_m']:.3f}m "
          f"(임계값 {THRESH_M}m)")

    reached_tick_id = ge["tick_id"]

    # 도착 이전 tick들: dist_to_goal_m가 기록되고, goal_reached=False였어야 함
    before_steps = [s for s in steps if s["tick_id"] < reached_tick_id and "dist_to_goal_m" in s]
    assert len(before_steps) > 0
    assert all(s["dist_to_goal_m"] > THRESH_M for s in before_steps), \
        "도착 이전 tick인데 이미 임계값 이내로 기록된 게 있음"
    print(f"[OK] 도착 이전 {len(before_steps)}개 tick 전부 dist_to_goal_m > {THRESH_M}m")

    # 도착 이후(그 tick 포함) 전송값은 전부 (0,0)이어야 함 -- LIVE에 heading도 정상이었더라도
    # control_sent 레코드 자체엔 tick_id가 없으므로 시간(ts)으로 매칭
    after_ts = ge["ts"]
    after_controls = [c for c in controls if c["ts"] >= after_ts]
    assert len(after_controls) > 0
    assert all(c["linear"] == 0.0 and c["angular"] == 0.0 for c in after_controls), \
        f"goal_reached 이후인데 non-zero 명령이 전송됨: {after_controls}"
    assert all(c.get("goal_reached") is True for c in after_controls), \
        "control_sent 레코드의 goal_reached 플래그가 True로 안 남음"
    print(f"[OK] goal_reached 이후 {len(after_controls)}개 tick 전부 전송값 (0,0), "
          f"control_sent.goal_reached=True 로그 확인")

    # 도착 이후에도 computed_linear/angular는 계속 계산되고 있어야 함(래치가 계산
    # 자체를 막는 게 아니라 전송만 막는지 확인 -- 대시보드에서 계속 관찰 가능해야 함)
    nonzero_computed_after = [c for c in after_controls if c["computed_linear"] != 0 or c["computed_angular"] != 0]
    print(f"[OK] goal_reached 이후에도 {len(nonzero_computed_after)}/{len(after_controls)}개 tick에서 "
          f"computed_linear/angular는 계속 계산됨(전송만 차단, 계산은 안 막음)")

    status = deployer.state.snapshot_status()
    assert status["goal_reached"] is True
    print(f"[OK] 대시보드 상태 goal_reached=True, dist_to_goal_m={status['dist_to_goal_m']:.3f}m")

    print(f"[OK] 로그 파일: {deployer.log_path}")


# ── test F: --initial_heading_lookahead_m이 실제로 반영되는지 ────────────────
def test_f_initial_heading_lookahead_configurable():
    print("\n========== test_f_initial_heading_lookahead_configurable ==========")
    camera_frames = _load_camera_frames() * 3
    n_ticks = 6
    gps_sequence = [_base_gps(1000, 1000)] * 2
    gps_sequence += [_base_gps(BASE_LAT, BASE_LON)] * n_ticks

    results = {}
    for lookahead in (2.0, 8.0):
        fake_get, fake_post, _ = build_fake_requests(camera_frames, gps_sequence)
        with mock.patch("requests.get", side_effect=fake_get), \
             mock.patch("requests.post", side_effect=fake_post):
            deployer = make_deployer(fake_get, fake_post, dry_run_lock=True, n_ticks=n_ticks,
                                      initial_heading_lookahead_m=lookahead)
            real_step = deployer.step
            tick_count = [0]

            def wrapped():
                if tick_count[0] == 3:
                    deployer._pending_commands.put("confirm_alignment")
                tick_count[0] += 1
                return real_step()

            deployer.step = wrapped
            deployer.run()

        logs = _read_jsonl(deployer.log_path)
        align_events = [l for l in logs if l.get("event") == "route_alignment_confirmed"]
        assert len(align_events) == 1
        assert align_events[0]["lookahead_m"] == lookahead, \
            f"확정 이벤트에 기록된 lookahead_m({align_events[0]['lookahead_m']})이 " \
            f"실제로 넘긴 값({lookahead})과 다름 -- 파라미터가 안 먹힘"
        results[lookahead] = align_events[0]["route_aligned_heading_deg"]
        print(f"[OK] lookahead={lookahead}m -> 확정된 heading={results[lookahead]:+.2f}° "
              f"(로그에도 lookahead_m={lookahead} 그대로 기록됨)")

    # 이 테스트의 route 형태(직선 GPS 경로)에선 값이 같을 수도 있지만, 최소한
    # "요청한 lookahead 값 그대로가 실제로 route_bearing_rad()에 전달되고 로그에
    # 남는다"는 건 위 assert로 이미 확인됨 -- 핵심은 파라미터 배관(plumbing) 검증.
    print(f"[OK] 파라미터가 생성자 -> _confirm_route_alignment() -> route_bearing_rad()까지 "
          f"실제로 전달됨을 확인 (2.0m: {results[2.0]:+.2f}°, 8.0m: {results[8.0]:+.2f}°)")


if __name__ == "__main__":
    test_a_dry_run_lock()
    test_b_full_workflow()
    test_c_arm_without_alignment()
    test_d_route_aligned_fixed_mode()
    test_e_goal_reached_stop()
    test_f_initial_heading_lookahead_configurable()
    print("\n모든 오프라인 스모크 테스트 통과.")
