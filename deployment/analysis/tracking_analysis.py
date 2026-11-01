"""
모델 출력 추종 분석 (2026-11-01, 테스트 C: 직진+회전 경로, 명령 변환 보정 끔/켬 비교).

질문: "모델이 원한 회전을 로봇이 실제로 했나?" — 모델/heading이 맞는지와는 별개로,
모델 출력 → 로봇 동작 단계만 본다.

매 LIVE 틱에서 세 가지 yaw rate(rad/s)를 비교:
  desired   : 모델(+클리핑)이 원한 회전 = step.angular_desired
              (override/cross-track 보정을 끈 실행이면 모델 출력 그대로)
  predicted : 실제 보낸 명령(control_sent.angular)에 실내 실측 모델을 적용한 값
              (|cmd|≤0.05 → 0, 그 외 1.66 × cmd) — 실내 모델이 실외에서도 맞는지 검증용
  actual    : 자이로 z(deg/s → rad/s, 틱 간 중복 샘플 제거)
로봇 반응 지연은 actual을 0~8틱 늦춰가며 desired와 상관이 최대인 값으로 맞춤.

구간 분류(모델과 무관하게 경로 기준): 로그의 route_bearing_deg가 ±3초 안에 20° 이상
바뀌면 "회전 구간", 아니면 "직진 구간".

실행: ~/miniconda3/envs/rover/bin/python deployment/analysis/tracking_analysis.py <log.jsonl> [<log2> ...]
출력: deployment/analysis/results/tracking_<run_id>/{summary.json, timeseries.png}
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ACT_GAIN = 1.66          # 실내 실측 rad/s per unit cmd (omnivla_edge_deploy.ACT_ANGULAR_GAIN_RAD_S)
ACT_MIN = 0.06
WHEEL_M_PER_REV = 0.329
TURN_WIN_S = 3.0
TURN_THRESH_DEG = 20.0


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def load(path):
    steps, sent, run_start = [], [], None
    for line in open(path):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = d.get("type")
        if t == "run_start":
            run_start = d
        elif t == "step":
            steps.append(d)
        elif t == "control_sent":
            sent.append(d)
    # step 다음에 오는 control_sent를 짝지음(같은 루프 반복)
    out, j = [], 0
    for s in steps:
        while j < len(sent) and sent[j]["ts"] < s["ts"]:
            j += 1
        c = sent[j] if j < len(sent) else None
        out.append((s, c))
    return run_start, out


def run_config(run_start):
    if not run_start:
        return {}
    cfg = dict(run_start)
    for k, v in run_start.items():
        if isinstance(v, dict):
            cfg.update(v)
    return cfg


def analyze(path):
    run_start, pairs = load(path)
    cfg = run_config(run_start)
    run_id = Path(path).stem.replace("deploy_", "")
    seen = set()
    rows = []
    for s, c in pairs:
        raw = s.get("frodobot_raw") or {}
        gz = []
        for g in raw.get("gyros") or []:
            if isinstance(g, list) and len(g) >= 4 and g[3] not in seen:
                seen.add(g[3])
                gz.append(g[2])
        rp = [r for r in (raw.get("rpms") or []) if isinstance(r, list) and len(r) >= 5]
        rows.append({
            "ts": s["ts"],
            "live": bool(c) and not c.get("dry_run", True),
            "desired": s.get("angular_desired"),
            "model_raw": s.get("angular_model_raw"),
            "sent": c.get("angular") if c else None,
            "sent_lin": c.get("linear") if c else None,
            "gyro": math.radians(float(np.mean(gz))) if gz else np.nan,
            "v": (float(np.mean([(r[0] + r[1] + r[2] + r[3]) / 4 for r in rp])) / 60 * WHEEL_M_PER_REV
                  if rp else np.nan),
            "route_bearing": s.get("route_bearing_deg"),
            "heading": s.get("smoothed_heading_deg"),
            "heading_src": s.get("map_heading_source"),
            "override": s.get("override_reason"),
            "lat": s.get("lat"), "lon": s.get("lon"),
            "dist_to_goal": s.get("dist_to_goal_m"),
        })
    live = [r for r in rows if r["live"] and r["desired"] is not None and r["sent"] is not None]
    if len(live) < 10:
        return run_id, {"error": f"LIVE 틱 {len(live)}개 — 분석 불가"}, rows, None
    n = len(live)
    ts = np.array([r["ts"] for r in live])
    des = np.array([r["desired"] for r in live], float)
    sent = np.array([r["sent"] for r in live], float)
    pred = np.where(np.abs(sent) <= 0.05 + 1e-9, 0.0, ACT_GAIN * sent)
    gyro = np.array([r["gyro"] for r in live], float)
    v = np.array([r["v"] for r in live], float)

    # 반응 지연: gyro[i+lag] vs desired[i]
    def lagged(x, lag):
        y = np.full(n, np.nan)
        if lag < n:
            y[:n - lag] = x[lag:]
        return y

    best_lag, best_r, by_lag = 0, -2, {}
    for lag in range(0, 9):
        g = lagged(gyro, lag)
        ok = np.isfinite(g)
        if ok.sum() >= 10 and np.std(des[ok]) > 1e-6 and np.std(g[ok]) > 1e-6:
            r = float(np.corrcoef(des[ok], g[ok])[0, 1])
            by_lag[lag] = round(r, 3)
            if r > best_r:
                best_r, best_lag = r, lag
    act = lagged(gyro, best_lag)

    # 회전 구간(경로 기준)
    rb = np.array([r["route_bearing"] if r["route_bearing"] is not None else np.nan for r in live], float)
    turn = np.zeros(n, bool)
    for i in range(n):
        j0 = np.searchsorted(ts, ts[i] - TURN_WIN_S)
        j1 = min(np.searchsorted(ts, ts[i] + TURN_WIN_S), n - 1)
        if np.isfinite(rb[j0]) and np.isfinite(rb[j1]) and abs(wrap_deg(rb[j1] - rb[j0])) >= TURN_THRESH_DEG:
            turn[i] = True

    dt = np.clip(np.diff(ts, prepend=ts[0]), 0, 2.0)

    def seg_stats(mask, name):
        ok = mask & np.isfinite(act)
        if ok.sum() < 5:
            return {"segment": name, "n": int(ok.sum())}
        d, a, p = des[ok], act[ok], pred[ok]
        slope = float(np.sum(d * a) / np.sum(d * d)) if np.sum(d * d) > 1e-9 else None
        return {
            "segment": name, "n": int(ok.sum()),
            "desired_mean_rad_s": float(d.mean()), "actual_mean_rad_s": float(a.mean()),
            "bias_actual_minus_desired": float((a - d).mean()),
            "gain_actual_over_desired": slope,
            "corr": float(np.corrcoef(d, a)[0, 1]) if np.std(d) > 1e-6 and np.std(a) > 1e-6 else None,
            "yaw_desired_deg": float(np.degrees(np.sum(d * dt[ok]))),
            "yaw_actual_deg": float(np.degrees(np.sum(a * dt[ok]))),
            "yaw_predicted_from_cmd_deg": float(np.degrees(np.sum(p * dt[ok]))),
            "share_desired_in_deadband": float(np.mean(np.abs(d) / ACT_GAIN <= 0.05)),
            "share_sent_in_deadband": float(np.mean(np.abs(sent[ok]) <= 0.05 + 1e-9)),
        }

    # 실외 게인 검증: |sent|≥0.06인 틱에서 gyro(lag 적용) / sent
    big = (np.abs(sent) >= ACT_MIN - 1e-9) & np.isfinite(act)
    outdoor_gain = (float(np.sum(sent[big] * act[big]) / np.sum(sent[big] ** 2))
                    if big.sum() >= 5 else None)

    summary = {
        "run_id": run_id,
        "actuation_calibration": cfg.get("enable_actuation_calibration"),
        "heading_mode": cfg.get("heading_mode"), "route_mode": cfg.get("route_mode"),
        "waypoint_target_step": cfg.get("waypoint_target_step"),
        "n_live_ticks": n, "n_turn_ticks": int(turn.sum()),
        "response_lag_ticks": best_lag, "corr_by_lag": by_lag,
        "mean_speed_mps": float(np.nanmean(v)),
        "overrides_seen": sorted({r["override"] for r in live if r["override"]}),
        "heading_sources": sorted({r["heading_src"] for r in live if r["heading_src"]}),
        "all": seg_stats(np.ones(n, bool), "all"),
        "straight": seg_stats(~turn, "straight"),
        "turn": seg_stats(turn, "turn"),
        "outdoor_gain_rad_s_per_cmd": outdoor_gain,
        "indoor_gain_rad_s_per_cmd": ACT_GAIN,
        "n_ticks_sent_ge_0.06": int(big.sum()),
    }
    return run_id, summary, rows, dict(ts=ts, des=des, pred=pred, act=act, turn=turn, rb=rb,
                                        heading=np.array([r["heading"] if r["heading"] is not None
                                                          else np.nan for r in live], float),
                                        sent=sent)


def plot(out_dir, run_id, series, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = series["ts"] - series["ts"][0]
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax in axs:
        for i in np.where(series["turn"])[0]:
            ax.axvspan(t[i] - 0.17, t[i] + 0.17, color="#ffe0b2", lw=0)
        ax.grid(alpha=0.3)
    ax = axs[0]
    ax.plot(t, series["des"], "k-", lw=1.5, label="desired (model)")
    ax.plot(t, series["pred"], color="#1f77b4", lw=1, label="predicted from sent cmd (indoor model)")
    ax.plot(t, series["act"], color="#d62728", lw=1, label=f"actual gyro (lag {summary['response_lag_ticks']} ticks)")
    ax.set_ylabel("yaw rate (rad/s, +left)")
    ax.legend(fontsize=8)
    ax.set_title(f"{run_id}  calibration={summary['actuation_calibration']}  (orange = route turn zone)")
    ax = axs[1]
    ax.step(t, series["sent"], where="post", color="#2ca02c", lw=1)
    ax.axhspan(-0.05, 0.05, color="#dddddd", alpha=0.6, label="deadband |cmd|≤0.05")
    ax.set_ylabel("sent angular cmd")
    ax.legend(fontsize=8)
    ax = axs[2]
    dt = np.clip(np.diff(series["ts"], prepend=series["ts"][0]), 0, 2.0)
    ok = np.isfinite(series["act"])
    ax.plot(t, np.degrees(np.cumsum(series["des"] * dt)), "k-", label="∫desired")
    ax.plot(t, np.degrees(np.cumsum(np.where(ok, series["act"], 0) * dt)), color="#d62728", label="∫actual")
    ax.plot(t, np.degrees(np.cumsum(series["pred"] * dt)), color="#1f77b4", label="∫predicted")
    ax.set_ylabel("cumulative yaw (deg)")
    ax.set_xlabel("s since LIVE")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "timeseries.png", dpi=110)
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        run_id, summary, rows, series = analyze(p)
        out_dir = REPO_ROOT / "deployment/analysis/results" / f"tracking_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        if series is not None:
            plot(out_dir, run_id, series, summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
