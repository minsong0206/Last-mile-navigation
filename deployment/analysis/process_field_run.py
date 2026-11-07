"""실기 테스트 1회분 사후처리 — 영상 3개 점검 + 로그 지표 추출 (2026-11-07).

배포 머신에서 run 하나가 끝나면 아래 3개가 자동 생성된다:
  - deployment/logs/<날짜>/deploy_<run_id>.jsonl          (틱별 로그)
  - deployment/logs/<...>/frames/<run_id>/ctx_*.jpg,map_*.png → compile_videos()가
    ctx_<run_id>.mp4(카메라) / map_<run_id>.mp4(모델입력 지도)로 자동 컴파일
  - deployment/logs/<...>/dashboard_<run_id>.mp4           (대시보드, 지도+예측 오버레이)

이 스크립트는 (1) 그 3개 영상이 실제로 재생 가능한지 ffprobe로 점검하고, 깨졌으면(크래시로
moov 소실 등) frames에서 Notion 안전 포맷(H.264 fMP4)으로 재인코딩을 시도한다. (2) 로그에서
실기 테스트 판정 지표(완주도/크로스트랙/heading 오차/좌우 명령 비율/모델 원함 vs 자이로 실측/
GPS 갱신 간격/동결 비율)를 뽑아 **아티팩트에 그대로 붙일 수 있는 마크다운 블록**으로 출력한다.

영상 자체는 claude.ai 아티팩트(HTML)에 못 넣으므로, 이 스크립트는 영상 "파일 경로"만 알려주고
지표 표를 만든다 — 영상은 사용자가 Notion/슬라이드에 직접 올린다.

실행:
  python3 deployment/analysis/process_field_run.py <run_id>
  python3 deployment/analysis/process_field_run.py --latest      # 가장 최근 run
"""
import argparse
import glob
import json
import math
import os
import statistics as st
import subprocess
from collections import Counter

RAD2DEG = 180.0 / math.pi
# 실외 실측(2026-11-06, outdoor_actuation_1106.py): 데드밴드 0.055, 게인 1.165 rad/s/cmd.
DEADBAND = 0.055


# ───────────────────────── 파일 찾기 ─────────────────────────
def find_run(run_id):
    """run_id로 log/frames/dashboard 경로를 찾는다. run_id=None이면 가장 최근 로그."""
    logs = sorted(glob.glob("deployment/logs/**/deploy_*.jsonl", recursive=True))
    if run_id is None:
        if not logs:
            raise SystemExit("deploy_*.jsonl 로그를 못 찾음.")
        log = logs[-1]
        run_id = os.path.basename(log)[len("deploy_"):-len(".jsonl")]
    else:
        cand = [p for p in logs if f"deploy_{run_id}.jsonl" in p]
        if not cand:
            raise SystemExit(f"deploy_{run_id}.jsonl 를 못 찾음. 있는 로그:\n  " +
                             "\n  ".join(os.path.basename(p) for p in logs[-8:]))
        log = cand[0]
    base = os.path.dirname(log)
    paths = {
        "run_id": run_id,
        "log": log,
        "frames_dir": os.path.join(base, "frames", run_id),
        "dashboard": os.path.join(base, f"dashboard_{run_id}.mp4"),
        # deploy의 compile_videos()는 frames/<run_id>/ 안에 아래 이름으로 저장함
        "ctx_video": os.path.join(base, "frames", run_id, "context_video.mp4"),
        "map_video": os.path.join(base, "frames", run_id, "map_video.mp4"),
    }
    return paths


# ───────────────────────── 영상 점검/복구 ─────────────────────────
def probe_ok(path):
    """ffprobe로 재생 가능 여부 + 길이(초). 없거나 깨졌으면 (False, 사유)."""
    if not os.path.exists(path):
        return False, "파일 없음"
    if os.path.getsize(path) < 1024:
        return False, f"{os.path.getsize(path)}바이트 — 껍데기(크래시로 소실 추정)"
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nk=1:nw=1", path],
            capture_output=True, text=True, timeout=30)
        dur = float(out.stdout.strip() or 0)
        return (dur > 0.1, f"{dur:.1f}s") if dur > 0.1 else (False, "길이 0")
    except Exception as e:
        return False, f"ffprobe 실패: {e}"


def recompile_from_frames(frames_dir, pattern, out_path, fps):
    """frames의 ctx_*.jpg / map_*.png 시퀀스에서 H.264 fMP4로 재인코딩(복구용)."""
    files = sorted(glob.glob(os.path.join(frames_dir, pattern)))
    if not files:
        return False, f"{pattern} 프레임 없음"
    start = int("".join(filter(str.isdigit, os.path.basename(files[0]))) or 0)
    ext = pattern.split(".")[-1]
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-start_number", str(start), "-framerate", f"{fps:.3f}",
             "-i", os.path.join(frames_dir, pattern.replace("*", "%06d")),
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+frag_keyframe+empty_moov+default_base_moof", "-g", "5",
             out_path],
            capture_output=True, timeout=120)
        if proc.returncode != 0:
            return False, f"ffmpeg code={proc.returncode}"
        return True, f"{len(files)}프레임 → {os.path.basename(out_path)} 복구"
    except FileNotFoundError:
        return False, "ffmpeg 없음"
    except Exception as e:
        return False, str(e)


def check_videos(paths):
    results = {}
    # ctx/map 은 모델 동작 주기(약 3Hz)로 저장 — compile_videos 기본 fps와 동일 가정
    for label, key, pattern, fps in [
        ("카메라(ctx)", "ctx_video", "ctx_*.jpg", 3.0),
        ("모델입력지도(map)", "map_video", "map_*.png", 3.0),
    ]:
        ok, info = probe_ok(paths[key])
        if not ok:
            rok, rinfo = recompile_from_frames(paths["frames_dir"], pattern, paths[key], fps)
            info = f"원본 불량({info}) → " + ("복구 성공: " + rinfo if rok else "복구 실패: " + rinfo)
            ok = rok
        results[label] = (ok, info, paths[key])
    ok, info = probe_ok(paths["dashboard"])
    if not ok:
        info += " (대시보드는 frames가 없어 복구 불가 — 다음 run부터 fMP4로 저장됨)"
    results["대시보드(dashboard)"] = (ok, info, paths["dashboard"])
    return results


# ───────────────────────── 지표 추출 ─────────────────────────
def _stat(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"median": st.median(xs), "p90": sorted(xs)[min(len(xs) - 1, int(len(xs) * 0.9))],
            "mean": st.mean(xs), "n": len(xs)}


def extract_metrics(log_path):
    L = [json.loads(x) for x in open(log_path)]
    steps = [l for l in L if l.get("type") == "step"]
    live = [l for l in steps if l.get("control_stage") == "LIVE"]
    m = {"n_step": len(steps), "n_live": len(live)}
    if not live:
        m["note"] = "LIVE 틱이 없음(GO LIVE 안 됨 또는 로그 이상)."
        return m

    ts = [l["ts"] for l in live if "ts" in l]
    m["duration_s"] = round(ts[-1] - ts[0], 1) if len(ts) >= 2 else None

    d = [l.get("dist_to_goal_m") for l in live if l.get("dist_to_goal_m") is not None]
    if d:
        m["dist_goal_start"] = round(d[0], 1)
        m["dist_goal_end"] = round(d[-1], 1)
        m["dist_goal_min"] = round(min(d), 1)
        m["progress_m"] = round(d[0] - min(d), 1)

    m["cross_track_abs"] = _stat([abs(l["cross_track_m"]) for l in live
                                  if l.get("cross_track_m") is not None])
    m["heading_diff_abs"] = _stat([abs(l["heading_diff_deg"]) for l in live
                                   if l.get("heading_diff_deg") is not None])

    # 좌우 명령 비율 (데드밴드 밖만 회전으로 카운트; +각속도=좌, -=우)
    ang = [l.get("angular") for l in live if l.get("angular") is not None]
    left = sum(1 for a in ang if a > DEADBAND)
    right = sum(1 for a in ang if a < -DEADBAND)
    dead = sum(1 for a in ang if abs(a) <= DEADBAND)
    m["cmd"] = {"n": len(ang), "left": left, "right": right, "deadband": dead,
                "mean": round(st.mean(ang), 4) if ang else None,
                "deadband_pct": round(dead / len(ang) * 100) if ang else None,
                "lr_ratio": (f"{left}:{right}" if right else f"{left}:0")}

    # 모델이 원한 회전(명령 angular, rad/s) vs 자이로 실측(gyro z, deg/s)
    pairs = []
    for l in live:
        raw = l.get("frodobot_raw") or {}
        gz = [g[2] for g in (raw.get("gyros") or []) if len(g) >= 3]
        if gz and l.get("angular") is not None:
            pairs.append((l["angular"], st.mean(gz)))
    if pairs:
        turn = [(a, g) for a, g in pairs if abs(a) > DEADBAND]
        agree = sum(1 for a, g in turn if (a > 0) == (g > 0))
        m["cmd_vs_gyro"] = {
            "n_with_gyro": len(pairs),
            "n_turn_cmd": len(turn),
            "sign_agree": f"{agree}/{len(turn)}" if turn else "회전 명령 없음",
            "gain_deg_s_per_cmd": (round(sum(g * a for a, g in turn) / sum(a * a for a, g in turn), 1)
                                   if turn and sum(a * a for a, g in turn) else None)}

    # GPS 갱신 간격 (lat/lon 실제 변화 시점 사이 Δt) + 동결 비율
    gts, glat, glon = [], None, None
    intervals, last_t = [], None
    for l in live:
        raw = l.get("frodobot_raw") or {}
        la, lo, t = raw.get("latitude"), raw.get("longitude"), l.get("ts")
        if la is None or lo is None or t is None:
            continue
        if la != glat or lo != glon:
            if last_t is not None:
                intervals.append(t - last_t)
            last_t, glat, glon = t, la, lo
    m["gps_interval_s"] = _stat(intervals)
    frozen = [l for l in live if l.get("gps_frozen")]
    m["gps_frozen_pct"] = round(len(frozen) / len(live) * 100) if live else None
    m["heading_source"] = dict(Counter(l.get("map_heading_source") for l in live))

    # GPS 수신 품질 (한 틱 표본)
    raw0 = next((l.get("frodobot_raw") for l in live if l.get("frodobot_raw")), {}) or {}
    m["gps_quality"] = {k: raw0.get(k) for k in
                        ("hdop", "fix_quality", "gps_signal", "signal_level")}
    return m


# ───────────────────────── 출력 ─────────────────────────
def fmt_stat(s, unit=""):
    if not s:
        return "—"
    return f"중앙 {s['median']:.2f}{unit} / p90 {s['p90']:.2f}{unit} / 평균 {s['mean']:.2f}{unit} (n={s['n']})"


def print_report(paths, videos, m):
    rid = paths["run_id"]
    print("\n" + "=" * 70)
    print(f"# 실기 테스트 사후처리 — run {rid}")
    print("=" * 70)

    print("\n## 영상 3개 상태 (Notion 업로드용 파일 경로)")
    for label, (ok, info, path) in videos.items():
        print(f"  [{'OK ' if ok else '불량'}] {label}: {info}")
        print(f"        {path}")

    print("\n## 로그 지표 (아래 블록을 아티팩트 해당 테스트 칸에 붙여넣기)")
    print("```")
    print(f"LIVE 틱 {m.get('n_live')}개 / 전체 {m.get('n_step')}개, "
          f"주행시간 {m.get('duration_s')}s")
    if "dist_goal_start" in m:
        print(f"완주도: 목표까지 {m['dist_goal_start']}m → 최소 {m['dist_goal_min']}m "
              f"(끝 {m['dist_goal_end']}m), 목표쪽 진행 {m['progress_m']}m")
    print(f"크로스트랙 |오차|: {fmt_stat(m.get('cross_track_abs'), 'm')}")
    print(f"heading 오차 |diff|: {fmt_stat(m.get('heading_diff_abs'), '°')}")
    c = m.get("cmd", {})
    if c:
        print(f"조향 명령: 좌:우 = {c.get('lr_ratio')}, 데드밴드(|a|≤{DEADBAND}) "
              f"{c.get('deadband')}개({c.get('deadband_pct')}%), 평균 angular {c.get('mean')}")
    cg = m.get("cmd_vs_gyro")
    if cg:
        print(f"모델 원함 vs 자이로 실측: 회전명령 {cg['n_turn_cmd']}틱 중 부호일치 "
              f"{cg['sign_agree']}, 실측게인 {cg.get('gain_deg_s_per_cmd')} deg/s/cmd")
    print(f"GPS 갱신간격: {fmt_stat(m.get('gps_interval_s'), 's')}, "
          f"동결 틱 {m.get('gps_frozen_pct')}%")
    print(f"heading 소스 분포: {m.get('heading_source')}")
    print(f"GPS 품질(표본): {m.get('gps_quality')}")
    if m.get("note"):
        print(f"⚠ {m['note']}")
    print("```")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?", default=None, help="예: 20261107_143000")
    ap.add_argument("--latest", action="store_true", help="가장 최근 run 자동 선택")
    args = ap.parse_args()
    paths = find_run(None if args.latest else args.run_id)
    videos = check_videos(paths)
    metrics = extract_metrics(paths["log"])
    print_report(paths, videos, metrics)


if __name__ == "__main__":
    main()
