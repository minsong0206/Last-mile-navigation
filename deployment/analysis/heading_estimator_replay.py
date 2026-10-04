"""
heading 추정기(estimate_heading_from_track) 파라미터 오프라인 재생 비교.

배경(docs/experiment_log.md §1-9/§1-10): `auto` 모드에서 GPS 양자화 때문에
`gps_track` heading 추정이 -21.5°~+73.4°까지 틀어지고, 그게 지도 회전을 왜곡시켜
모델이 (내용 자체는 충실히 따라가며) 실제와 무관한 큰 좌/우 angular를 내는 게
확인됨. 사용자/Harness 2가 제안한 "gps_track 추정 자체를 더 보수적으로"(min_disp_m↑,
fast_disp_m↑, EMA alpha↓) 방향이 실제로 효과가 있는지, 실기기 테스트(배터리 소모)
전에 **이미 기록된 실제 run의 GPS 이력**으로 먼저 검증한다.

방법: `omnivla_edge_deploy.py`의 `estimate_heading_from_track()`/`gps_heading_readiness()`
+ 관련 상수를 **그대로 import**(수정 없음)해서, 실제 step 레코드의 lat/lon/ts/
gps_fix_ok 시퀀스를 그대로 재생하며 production의 전체 heading 파이프라인(rate-limit
clamp → EMA → route_bearing 강제 재동기화 → past_track 초기화)을 동일하게 재구현한다.
route_bearing_rad_now은 매 tick 재계산하지 않고 **로그에 이미 저장된
`route_bearing_deg`를 그대로 사용**(위치만 의존하고 heading 추정과는 무관한 순수
기하값이라 재계산 불필요, OSRM 재호출도 필요 없음 — 로봇 안 움직임).

현재 파라미터로 재생한 결과가 로그에 실제 기록된 `smoothed_heading_deg`/
`map_heading_source`/`heading_route_diff_deg`와 **정확히 일치하는지 먼저 검증**
(불일치하면 재생 로직에 버그가 있다는 뜻이므로 그 결과는 못 믿음 — 아래
validate_replay() 참고).

실행: python3 deployment/analysis/heading_estimator_replay.py
"""
import sys
import math
import json
from pathlib import Path
from collections import namedtuple

REPO_ROOT = Path("/home/moai/Last-mile-navigation")
sys.path.insert(0, str(REPO_ROOT / "deployment"))

from omnivla_edge_deploy import (  # noqa: E402
    estimate_heading_from_track,
    MAX_W,
    HEADING_EMA_ALPHA as PROD_EMA_ALPHA,
    HEADING_RATE_LIMIT_MARGIN as PROD_RATE_LIMIT_MARGIN,
    HEADING_ROUTE_CORRECTION_THRESHOLD_DEG as PROD_CORR_THRESH_DEG,
    HEADING_ROUTE_CORRECTION_PERSIST_TICKS as PROD_PERSIST_TICKS,
)

Config = namedtuple("Config", ["name", "min_disp_m", "fast_disp_m", "fast_disagree_deg",
                                "ema_alpha", "rate_limit_margin",
                                "corr_thresh_deg", "persist_ticks"])

CURRENT = Config("CURRENT(production)", 1.5, 0.5, 35.0,
                  PROD_EMA_ALPHA, PROD_RATE_LIMIT_MARGIN,
                  PROD_CORR_THRESH_DEG, PROD_PERSIST_TICKS)


def load_run(run_id):
    path = REPO_ROOT / "deployment" / "logs" / f"deploy_{run_id}.jsonl"
    steps, route_align = [], None
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "step":
                steps.append(r)
            elif r.get("type") == "event" and r.get("event") == "route_alignment_confirmed":
                route_align = r
    return steps, route_align


def replay(steps, route_align, cfg: Config):
    """production omnivla_edge_deploy.py OmniVLAEdgeDeployment.step()의 heading 관련
    블록(약 909-1069줄)을 동일하게 재구현. 반환: tick_id -> dict(source, smoothed_deg, hrd_diff)"""
    route_aligned_heading_rad = math.radians(route_align["route_aligned_heading_deg"])

    past_track = []
    heading_ema_vec = None
    prev_ts = None
    divergence_ticks = 0
    out = {}

    started = False
    for s in steps:
        if not started:
            if route_align is not None and s["ts"] >= route_align["ts"]:
                started = True
            else:
                continue  # ARM 이전(imu_fallback 구간)은 분석 대상 아님 — production도 이 구간은 heading 안 씀

        ts = s["ts"]
        if s.get("gps_fix_ok", True) and s.get("lat") not in (None, 1000):
            past_track.append((s["lat"], s["lon"]))
            if len(past_track) > 200:
                past_track = past_track[-200:]

        gps_heading_rad = estimate_heading_from_track(
            past_track, min_disp_m=cfg.min_disp_m,
            fast_disp_m=cfg.fast_disp_m, fast_disagree_deg=cfg.fast_disagree_deg)

        route_bearing_deg = s.get("route_bearing_deg")
        route_bearing_rad_now = math.radians(route_bearing_deg) if route_bearing_deg is not None else None

        if gps_heading_rad is not None:
            if heading_ema_vec is not None and prev_ts is not None:
                dt_s = ts - prev_ts
                cur_angle = math.atan2(heading_ema_vec.imag, heading_ema_vec.real)
                raw_diff = (gps_heading_rad - cur_angle + math.pi) % (2 * math.pi) - math.pi
                max_dtheta = MAX_W * max(dt_s, 0.0) * cfg.rate_limit_margin
                if abs(raw_diff) > max_dtheta:
                    gps_heading_rad = cur_angle + math.copysign(max_dtheta, raw_diff)
            new_vec = complex(math.cos(gps_heading_rad), math.sin(gps_heading_rad))
            if heading_ema_vec is None:
                heading_ema_vec = new_vec
            else:
                heading_ema_vec = (1 - cfg.ema_alpha) * heading_ema_vec + cfg.ema_alpha * new_vec
            map_heading_rad = math.atan2(heading_ema_vec.imag, heading_ema_vec.real)
            source = "gps_track"
        elif heading_ema_vec is not None:
            map_heading_rad = math.atan2(heading_ema_vec.imag, heading_ema_vec.real)
            source = "gps_ema_hold"
        elif route_aligned_heading_rad is not None:
            map_heading_rad = route_aligned_heading_rad
            source = "route_aligned"
        else:
            map_heading_rad = 0.0
            source = "imu_fallback"

        if source in ("gps_track", "gps_ema_hold") and route_bearing_rad_now is not None:
            diff_deg = (math.degrees(map_heading_rad - route_bearing_rad_now) + 180) % 360 - 180
            if abs(diff_deg) >= cfg.corr_thresh_deg:
                divergence_ticks += 1
            else:
                divergence_ticks = 0
            if divergence_ticks >= cfg.persist_ticks:
                map_heading_rad = route_bearing_rad_now
                heading_ema_vec = complex(math.cos(map_heading_rad), math.sin(map_heading_rad))
                past_track = []
                divergence_ticks = 0
                source = "route_corrected"

        prev_ts = ts
        hrd_diff = None
        if route_bearing_rad_now is not None and source != "route_corrected":
            hrd_diff = (math.degrees(map_heading_rad - route_bearing_rad_now) + 180) % 360 - 180
        elif source == "route_corrected":
            hrd_diff = 0.0
        out[s["tick_id"]] = dict(source=source, smoothed_deg=math.degrees(map_heading_rad), hrd_diff=hrd_diff)
    return out


def validate_replay(steps, route_align, run_id):
    """CURRENT 설정으로 재생한 결과가 로그 실측과 정확히 일치하는지 확인 (±0.5° 허용)."""
    result = replay(steps, route_align, CURRENT)
    n_checked, n_mismatch = 0, 0
    for s in steps:
        if s.get("control_stage") != "LIVE":
            continue
        logged_source = s.get("map_heading_source")
        logged_deg = s.get("smoothed_heading_deg")
        r = result.get(s["tick_id"])
        if r is None or logged_deg is None:
            continue
        n_checked += 1
        d = abs(((r["smoothed_deg"] - logged_deg + 180) % 360) - 180)
        if d > 0.5 or (logged_source in ("gps_track", "gps_ema_hold", "route_corrected")
                       and r["source"] != logged_source
                       and not (logged_source == "gps_ema_hold" and r["source"] == "gps_track")):
            n_mismatch += 1
            if n_mismatch <= 5:
                print(f"  [MISMATCH] {run_id} tick={s['tick_id']} logged=({logged_source},{logged_deg:.1f}) "
                      f"replay=({r['source']},{r['smoothed_deg']:.1f})")
    print(f"[validate] {run_id}: checked={n_checked} mismatch={n_mismatch}")
    return n_mismatch == 0


def summarize(result, steps, label):
    live_tids = {s["tick_id"] for s in steps if s.get("control_stage") == "LIVE"}
    diffs = [r["hrd_diff"] for tid, r in result.items() if tid in live_tids and r["hrd_diff"] is not None]
    n_corrections = sum(1 for r in result.values() if r["source"] == "route_corrected")
    max_abs = max((abs(d) for d in diffs), default=0.0)
    n_over20 = sum(1 for d in diffs if abs(d) >= 20)
    n_over45 = sum(1 for d in diffs if abs(d) >= 45)
    mean_abs = sum(abs(d) for d in diffs) / len(diffs) if diffs else 0.0
    print(f"  {label:28s}  max|diff|={max_abs:6.1f}°  mean|diff|={mean_abs:5.1f}°  "
          f"ticks>=20°={n_over20:3d}  ticks>=45°={n_over45:3d}  forced_corrections={n_corrections}")


CANDIDATES = [
    CURRENT,
    Config("min_disp=2.5, fast=1.0", 2.5, 1.0, 35.0, PROD_EMA_ALPHA, PROD_RATE_LIMIT_MARGIN,
           PROD_CORR_THRESH_DEG, PROD_PERSIST_TICKS),
    Config("min_disp=3.0, fast=1.5", 3.0, 1.5, 35.0, PROD_EMA_ALPHA, PROD_RATE_LIMIT_MARGIN,
           PROD_CORR_THRESH_DEG, PROD_PERSIST_TICKS),
    Config("min_disp=1.5, fast_off(99)", 1.5, 99.0, 35.0, PROD_EMA_ALPHA, PROD_RATE_LIMIT_MARGIN,
           PROD_CORR_THRESH_DEG, PROD_PERSIST_TICKS),
    Config("CURRENT min_disp + EMA alpha=0.15", 1.5, 0.5, 35.0, 0.15, PROD_RATE_LIMIT_MARGIN,
           PROD_CORR_THRESH_DEG, PROD_PERSIST_TICKS),
    Config("min_disp=2.5,fast_off + EMA=0.15", 2.5, 99.0, 35.0, 0.15, PROD_RATE_LIMIT_MARGIN,
           PROD_CORR_THRESH_DEG, PROD_PERSIST_TICKS),
    Config("min_disp=4.0, fast_off(99)", 4.0, 99.0, 35.0, PROD_EMA_ALPHA, PROD_RATE_LIMIT_MARGIN,
           PROD_CORR_THRESH_DEG, PROD_PERSIST_TICKS),
]

if __name__ == "__main__":
    for run_id in ["20261004_105158", "20261004_104301"]:
        print(f"\n{'='*70}\nrun: {run_id}\n{'='*70}")
        steps, route_align = load_run(run_id)
        ok = validate_replay(steps, route_align, run_id)
        print(f"[validate] {'PASS — 재생 로직이 production과 정확히 일치함' if ok else 'FAIL — 아래 결과는 참고만'}")
        print()
        for cfg in CANDIDATES:
            result = replay(steps, route_align, cfg)
            summarize(result, steps, cfg.name)
