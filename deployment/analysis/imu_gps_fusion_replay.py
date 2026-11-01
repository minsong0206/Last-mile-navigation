"""
IMU/GPS 융합 오프라인 리플레이 — Phase A: 센서 감사(sensor audit).

배경(2026-11-01 세션): 원거리 주행의 병목은 위치/heading 추정(GPS 동결 최대 82%,
1.5m 창 heading 지연·흔들림)이라는 결론. EKF 등 융합 필터를 붙이기 전에 필터 입력
(자이로, 바퀴 rpm, GPS)이 실제로 쓸 만한지 기존 로그만으로 숫자로 확인한다.
예전 자이로 융합 시도는 "자이로 버스트가 틱 간격의 16~25%만 커버, 부호 일치 51~58%"로
포기했었음(omnivla_edge_deploy.py USE_GYRO_FUSION 주석) — 이번엔 rpm까지 포함해 재감사.

로봇을 움직이지 않음. 배포 코드는 import도 하지 않음(순수 로그 분석).

입력: deployment/logs/deploy_*.jsonl (0918~1004)
      deployment/logs/hf_20261011/logs/deploy_*.jsonl (HF minsonganingee/frodobot-deploy-logs-20261011)
출력: deployment/analysis/results/fusion_replay_<YYYYMMDD>/audit_*.{json,png}

실행: ~/miniconda3/envs/rover/bin/python deployment/analysis/imu_gps_fusion_replay.py audit

주의(가정):
- control_sent.angular = 실제 전송값(trim/dither 적용 후). 로봇 펌웨어가 0.01 단위로
  0 쪽 절삭하므로(docs/1011.md §1) 분석에는 절삭 후 "유효 명령"을 쓴다.
- 센서 샘플 timestamp(로봇 시계)와 호스트 ts는 시계가 다름 → 샘플은 "처음 도착한 틱"에
  귀속시키고, 명령과의 시간 지연(lag)은 0~8틱을 다 시도해 상관이 가장 큰 값을 보고한다.
- rpms 각 샘플 = [ch0, ch1, ch2, ch3, ts]. 어느 채널이 좌/우인지 모름 → 데이터로 판별.
"""
import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_GLOBS = [
    str(REPO_ROOT / "deployment/logs/deploy_*.jsonl"),
    str(REPO_ROOT / "deployment/logs/hf_20261011/logs/deploy_*.jsonl"),
]
MIN_MOVING_TICKS = 30
MOVING_SPEED_MIN = 0.1      # frodobot_raw.speed (단위 불명, 정지 시 0)
ANGULAR_QUANTUM = 0.01      # 펌웨어 절삭 단위 (docs/1011.md §1)
SCALE_WIN_S = 15.0          # rpm→거리 스케일 추정 창 길이(초)

# rpm 채널 조합 후보: (좌측으로 가정한 채널들, 우측으로 가정한 채널들)
RPM_PAIRINGS = {
    "(0+2)-(1+3)": ((0, 2), (1, 3)),
    "(0+1)-(2+3)": ((0, 1), (2, 3)),
    "(0+3)-(1+2)": ((0, 3), (1, 2)),
}


# ─────────────────────────── 로딩 ───────────────────────────

def effective_angular(a):
    """펌웨어의 0.01 단위 0-쪽 절삭을 재현."""
    return math.trunc(round(a / ANGULAR_QUANTUM, 6)) * ANGULAR_QUANTUM


def load_run(path):
    """step 레코드마다 다음 control_sent를 짝지어 틱 리스트로 반환."""
    ticks, pending = [], None
    for line in open(path):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = d.get("type")
        if t == "step":
            raw = d.get("frodobot_raw")
            if not isinstance(raw, dict):
                pending = None
                continue
            pending = {
                "ts": float(d["ts"]),
                "lat": raw.get("latitude"), "lon": raw.get("longitude"),
                "speed": float(raw.get("speed") or 0.0),
                "gyros": raw.get("gyros") or [], "rpms": raw.get("rpms") or [],
                "gps_fix_ok": d.get("gps_fix_ok"),
                "sent_lin": 0.0, "sent_ang": 0.0, "live": False,
            }
            ticks.append(pending)
        elif t == "control_sent" and pending is not None:
            pending["sent_lin"] = float(d.get("linear") or 0.0)
            pending["sent_ang"] = float(d.get("angular") or 0.0)
            pending["live"] = not d.get("dry_run", True)
            pending = None
    for tk in ticks:
        tk["eff_ang"] = effective_angular(tk["sent_ang"])
        tk["moving"] = tk["speed"] > MOVING_SPEED_MIN
    return ticks


def dedupe_samples(ticks, key, n_vals):
    """틱 간 중복 샘플을 timestamp로 제거. 각 샘플을 처음 도착한 틱 인덱스에 귀속.
    반환: per-tick 새 샘플 배열 리스트, 전체 고유 샘플 timestamp 배열."""
    seen, per_tick, all_ts = set(), [], []
    for tk in ticks:
        new = []
        for s in tk[key]:
            if not isinstance(s, (list, tuple)) or len(s) < n_vals + 1:
                continue
            ts = round(float(s[n_vals]), 4)
            if ts in seen:
                continue
            seen.add(ts)
            new.append([float(v) for v in s[:n_vals]])
            all_ts.append(ts)
        per_tick.append(np.array(new) if new else np.zeros((0, n_vals)))
    return per_tick, np.array(sorted(all_ts))


def to_enu(lat, lon, lat0, lon0):
    x = (lon - lon0) * math.cos(math.radians(lat0)) * 111320.0
    y = (lat - lat0) * 110540.0
    return x, y


# ─────────────────────────── 지표 ───────────────────────────

def coverage_stats(sample_ts):
    if len(sample_ts) < 3:
        return None
    d = np.diff(sample_ts)
    period = float(np.median(d[d > 0])) if np.any(d > 0) else float("nan")
    span = float(sample_ts[-1] - sample_ts[0])
    return {
        "n_unique": int(len(sample_ts)),
        "sample_period_s": period,
        "coverage_ratio": float(min(1.0, len(sample_ts) * period / span)) if span > 0 else None,
        "gap_p50_s": float(np.percentile(d, 50)),
        "gap_p90_s": float(np.percentile(d, 90)),
        "gap_max_s": float(d.max()),
    }


def gps_freshness(ticks):
    """주행 틱 기준 GPS 갱신(값 변화) 통계와 동결 길이 분포(초)."""
    fresh_flags, freeze_lens, cur_start = [], [], None
    prev = None
    for tk in ticks:
        pos = (tk["lat"], tk["lon"])
        fresh = prev is not None and pos != prev
        prev = pos
        if not tk["moving"]:
            continue
        fresh_flags.append(fresh)
        if not fresh:
            if cur_start is None:
                cur_start = tk["ts"]
            cur_end = tk["ts"]
        elif cur_start is not None:
            freeze_lens.append(cur_end - cur_start)
            cur_start = None
    if not fresh_flags:
        return None
    fl = np.array(freeze_lens) if freeze_lens else np.zeros(1)
    return {
        "moving_ticks": len(fresh_flags),
        "frozen_ratio": float(1.0 - np.mean(fresh_flags)),
        "freeze_len_p50_s": float(np.percentile(fl, 50)),
        "freeze_len_p90_s": float(np.percentile(fl, 90)),
        "freeze_len_max_s": float(fl.max()),
        "freeze_lens_s": [float(v) for v in freeze_lens],
    }


def per_tick_means(per_tick, n_vals):
    out = np.full((len(per_tick), n_vals), np.nan)
    for i, a in enumerate(per_tick):
        if len(a):
            out[i] = a.mean(axis=0)
    return out


def best_lag_corr(x, cmd, mask, lags=tuple(range(0, 9))):
    """x[i]와 cmd[i-lag]의 상관. 반환 (best_lag, corr, n, {lag: corr})."""
    res = {}
    for lag in lags:
        xs, cs = [], []
        for i in range(lag, len(x)):
            if mask[i] and mask[i - lag] and np.isfinite(x[i]):
                xs.append(x[i])
                cs.append(cmd[i - lag])
        if len(xs) >= 20 and np.std(xs) > 0 and np.std(cs) > 0:
            res[lag] = (float(np.corrcoef(xs, cs)[0, 1]), len(xs))
    if not res:
        return None
    best = max(res, key=lambda k: abs(res[k][0]))
    return best, res[best][0], res[best][1], {k: v[0] for k, v in res.items()}


def gps_course_windows(ticks, min_disp_m=3.0):
    """GPS 갱신점만 써서 min_disp_m 이상 떨어진 연속점들로 course를 만들고, 연속 course
    두 개 사이의 heading 변화(rad)와 그 구간 틱 인덱스 범위를 반환."""
    lat0, lon0 = ticks[0]["lat"], ticks[0]["lon"]
    pts, prev = [], None
    for i, tk in enumerate(ticks):
        pos = (tk["lat"], tk["lon"])
        if pos != prev and tk["lat"] is not None and abs(tk["lat"]) <= 90:
            x, y = to_enu(tk["lat"], tk["lon"], lat0, lon0)
            pts.append((i, x, y))
        prev = pos
    anchors = [pts[0]] if pts else []
    for p in pts[1:]:
        if math.hypot(p[1] - anchors[-1][1], p[2] - anchors[-1][2]) >= min_disp_m:
            anchors.append(p)
    courses = []
    for a, b in zip(anchors, anchors[1:]):
        courses.append(((a[0] + b[0]) // 2, math.atan2(b[2] - a[2], b[1] - a[1])))
    wins = []
    for (i0, c0), (i1, c1) in zip(courses, courses[1:]):
        dc = (c1 - c0 + math.pi) % (2 * math.pi) - math.pi
        wins.append((i0, i1, dc))
    return wins, anchors


# ─────────────────────────── 감사 본체 ───────────────────────────

def audit(out_dir):
    files = sorted(set(f for g in LOG_GLOBS for f in glob.glob(g)))
    runs = []
    for f in files:
        ticks = load_run(f)
        if sum(tk["moving"] for tk in ticks) >= MIN_MOVING_TICKS:
            runs.append((Path(f).stem.replace("deploy_", ""), ticks))
    print(f"[load] 로그 {len(files)}개 중 주행 run {len(runs)}개 "
          f"(주행 틱 {sum(sum(t['moving'] for t in tk) for _, tk in runs)})")

    per_run = {}
    pool = {k: [] for k in ["cmd", "gyro", "rpm", "dt", "moving", "live", "speed"]}
    scale_pts, unit_pts = [], {"gyro_z": [], **{p: [] for p in RPM_PAIRINGS}}
    freeze_all = []

    for run_id, ticks in runs:
        g_tick, g_ts = dedupe_samples(ticks, "gyros", 3)
        r_tick, r_ts = dedupe_samples(ticks, "rpms", 4)
        gm = per_tick_means(g_tick, 3)
        rm = per_tick_means(r_tick, 4)
        ts = np.array([tk["ts"] for tk in ticks])
        dt = np.clip(np.diff(ts, prepend=ts[0]), 0, 2.0)
        moving = np.array([tk["moving"] for tk in ticks])
        live = np.array([tk["live"] for tk in ticks])
        cmd = np.array([tk["eff_ang"] for tk in ticks])
        speed = np.array([tk["speed"] for tk in ticks])

        gf = gps_freshness(ticks)
        if gf:
            freeze_all += gf.pop("freeze_lens_s")
        per_run[run_id] = {
            "ticks": len(ticks), "moving_ticks": int(moving.sum()), "live_ticks": int(live.sum()),
            "gyro_coverage": coverage_stats(g_ts), "rpm_coverage": coverage_stats(r_ts),
            "gyro_new_samples_per_tick": float(np.mean([len(a) for a in g_tick])),
            "rpm_new_samples_per_tick": float(np.mean([len(a) for a in r_tick])),
            "gps": gf,
        }
        for k, v in [("cmd", cmd), ("gyro", gm), ("rpm", rm), ("dt", dt),
                     ("moving", moving), ("live", live), ("speed", speed)]:
            pool[k].append(v)

        # rpm → 거리 스케일: GPS 갱신점 사이(≥SCALE_WIN_S) 구간의 GPS 직선거리 vs Σ(rpm/60·dt).
        # 짧은 창(5s)에서는 GPS 양자화 때문에 거리가 ~2m에 몰려 revs와 무상관이었음(1차 실행).
        # 창을 겹치지 않게 이어 붙이고, 직진(|cmd|≤0.08)·전 구간 주행인 창만 사용.
        lat0, lon0 = ticks[0]["lat"], ticks[0]["lon"]
        fresh_idx, prev = [], None
        for i, tk in enumerate(ticks):
            pos = (tk["lat"], tk["lon"])
            if pos != prev and tk["lat"] is not None and abs(tk["lat"]) <= 90:
                fresh_idx.append(i)
            prev = pos
        with np.errstate(all="ignore"):
            rpm_mean = np.nanmean(np.abs(rm), axis=1) if len(rm) else np.array([])
        rpm_rate = np.nan_to_num(rpm_mean, nan=0.0)
        a = 0
        for b in range(1, len(fresh_idx)):
            i0, i1 = fresh_idx[a], fresh_idx[b]
            if ts[i1] - ts[i0] < SCALE_WIN_S:
                continue
            seg = slice(i0 + 1, i1 + 1)
            ok_seg = np.all(moving[seg]) and np.all(np.abs(cmd[seg]) <= 0.08) \
                and np.mean(np.isfinite(rpm_mean[seg])) >= 0.8
            if ok_seg:
                x0, y0 = to_enu(ticks[i0]["lat"], ticks[i0]["lon"], lat0, lon0)
                x1, y1 = to_enu(ticks[i1]["lat"], ticks[i1]["lon"], lat0, lon0)
                dist = math.hypot(x1 - x0, y1 - y0)
                revs = float(np.sum(rpm_rate[seg] / 60.0 * dt[seg]))
                if revs > 1.0:
                    scale_pts.append((revs, dist))
            a = b

        # 단위/부호 판별: GPS course 변화 vs 그 구간 자이로·rpm차 적분
        wins, _ = gps_course_windows(ticks)
        for i0, i1, dc in wins:
            seg = slice(i0 + 1, i1 + 1)
            if abs(dc) < math.radians(20) or not np.all(moving[seg]):
                continue
            dur = dt[seg]
            gz = gm[seg, 2]
            ok = np.isfinite(gz)
            if ok.sum() >= max(3, 0.5 * len(gz)):
                unit_pts["gyro_z"].append((dc, float(np.nansum(np.where(ok, gz, 0) * dur)
                                                     * len(gz) / ok.sum())))
            for name, (L, R) in RPM_PAIRINGS.items():
                diff = rm[seg][:, list(L)].sum(1) - rm[seg][:, list(R)].sum(1)
                okr = np.isfinite(diff)
                if okr.sum() >= max(3, 0.5 * len(diff)):
                    unit_pts[name].append((dc, float(np.nansum(np.where(okr, diff, 0) * dur))))

    P = {k: np.concatenate(v) for k, v in pool.items()}
    mask = P["moving"] & P["live"]
    summary = {"n_runs": len(runs), "n_moving_live_ticks": int(mask.sum())}

    # 1) 커버리지
    def agg(field, sub):
        vals = [r[field][sub] for r in per_run.values() if r[field] and r[field][sub] is not None]
        return {"median": float(np.median(vals)), "p10": float(np.percentile(vals, 10)),
                "p90": float(np.percentile(vals, 90))} if vals else None
    summary["coverage"] = {
        "gyro_coverage_ratio": agg("gyro_coverage", "coverage_ratio"),
        "gyro_sample_period_s": agg("gyro_coverage", "sample_period_s"),
        "gyro_gap_p90_s": agg("gyro_coverage", "gap_p90_s"),
        "rpm_coverage_ratio": agg("rpm_coverage", "coverage_ratio"),
        "rpm_gap_p90_s": agg("rpm_coverage", "gap_p90_s"),
    }

    # 2) GPS
    fz = np.array(freeze_all) if freeze_all else np.zeros(1)
    fr = [r["gps"]["frozen_ratio"] for r in per_run.values() if r["gps"]]
    summary["gps"] = {
        "frozen_ratio_median_over_runs": float(np.median(fr)),
        "frozen_ratio_max": float(np.max(fr)),
        "freeze_len_s": {"p50": float(np.percentile(fz, 50)), "p90": float(np.percentile(fz, 90)),
                          "max": float(fz.max()), "n_episodes": len(freeze_all)},
    }

    # 3) 명령(유효 angular) vs 센서 상관 — 축/채널/부호/지연 판별
    corr = {}
    for ax in range(3):
        corr[f"gyro_axis{ax}"] = best_lag_corr(P["gyro"][:, ax], P["cmd"], mask)
    for name, (L, R) in RPM_PAIRINGS.items():
        diff = P["rpm"][:, list(L)].sum(1) - P["rpm"][:, list(R)].sum(1)
        corr[f"rpm_{name}"] = best_lag_corr(diff, P["cmd"], mask)
    summary["cmd_correlation"] = {k: (None if v is None else
                                      {"best_lag_ticks": v[0], "corr": v[1], "n": v[2],
                                       "by_lag": v[3]}) for k, v in corr.items()}

    # 부호 일치율: 의미 있는 회전 명령(|cmd|>=0.05)에서 gyro z 부호 == cmd 부호
    gz_res = corr["gyro_axis2"]
    if gz_res:
        lag = gz_res[0]
        sgn = np.sign(gz_res[1])
        agree = [np.sign(P["gyro"][i, 2]) * sgn == np.sign(P["cmd"][i - lag])
                 for i in range(lag, len(mask))
                 if mask[i] and mask[i - lag] and abs(P["cmd"][i - lag]) >= 0.05
                 and np.isfinite(P["gyro"][i, 2]) and P["gyro"][i, 2] != 0]
        summary["gyro_z_sign_agreement_turn_cmds"] = {"ratio": float(np.mean(agree)) if agree else None,
                                                       "n": len(agree)}

    # 4) 단위/스케일: Δcourse(rad) vs 센서 적분
    units = {}
    for k, pts in unit_pts.items():
        if len(pts) < 5:
            units[k] = {"n": len(pts)}
            continue
        dc, s = np.array(pts).T
        ratio = s / dc
        units[k] = {"n": len(pts), "sign_agreement": float(np.mean(np.sign(s) == np.sign(dc))),
                    "corr": float(np.corrcoef(dc, s)[0, 1]),
                    "median_ratio_sensor_per_rad": float(np.median(ratio))}
    summary["turn_window_units"] = units
    summary["turn_window_units_note"] = ("gyro_z ratio≈1 → rad/s, ≈57.3 → deg/s (|ratio| 기준). "
                                         "rpm 조합은 sign/corr가 가장 큰 쪽이 실제 좌우 배치.")

    # 5) rpm → 거리 스케일 (m/rev)
    if len(scale_pts) >= 5:
        revs, dist = np.array(scale_pts).T
        s_ls = float(np.sum(revs * dist) / np.sum(revs ** 2))
        per = dist / revs
        summary["rpm_distance_scale"] = {
            "n_segments": len(scale_pts), "window_s": SCALE_WIN_S,
            "corr_revs_vs_gps_dist": float(np.corrcoef(revs, dist)[0, 1]),
            "m_per_rev_ls": s_ls,
            "m_per_rev_median": float(np.median(per)),
            "m_per_rev_iqr": [float(np.percentile(per, 25)), float(np.percentile(per, 75))],
            "implied_wheel_radius_m": s_ls / (2 * math.pi),
        }
    else:
        summary["rpm_distance_scale"] = {"n_segments": len(scale_pts)}

    # speed 필드 vs rpm 평균
    rpm_mean_all = np.nanmean(np.abs(P["rpm"]), axis=1)
    ok = mask & np.isfinite(rpm_mean_all)
    if ok.sum() > 20:
        k = float(np.sum(rpm_mean_all[ok] * P["speed"][ok]) / np.sum(rpm_mean_all[ok] ** 2))
        summary["speed_field_vs_rpm"] = {"speed_per_rpm": k,
                                         "corr": float(np.corrcoef(rpm_mean_all[ok], P["speed"][ok])[0, 1])}

    # 6) 직진 좌우 rpm 비대칭: (L-R)/(L+R) vs 유효 명령 선형회귀 → 바퀴가 같아지는 명령값
    best_pair = max(RPM_PAIRINGS, key=lambda n: abs(corr[f"rpm_{n}"][1]) if corr[f"rpm_{n}"] else 0)
    L, R = RPM_PAIRINGS[best_pair]
    pr = corr[f"rpm_{best_pair}"]
    lag = pr[0] if pr else 0
    sign = np.sign(pr[1]) if pr else 1.0      # +면 L 그룹이 cmd>0(좌회전)에서 커짐
    xs, ys = [], []
    for i in range(lag, len(mask)):
        if not (mask[i] and mask[i - lag]) or abs(P["cmd"][i - lag]) > 0.15:
            continue
        l, r = P["rpm"][i, list(L)].sum(), P["rpm"][i, list(R)].sum()
        if np.isfinite(l) and np.isfinite(r) and (abs(l) + abs(r)) > 20:
            xs.append(P["cmd"][i - lag])
            ys.append((abs(l) - abs(r)) / (abs(l) + abs(r)))
    if len(xs) > 30:
        slope, icpt = np.polyfit(xs, ys, 1)
        summary["wheel_balance"] = {
            "pairing": best_pair, "lag_ticks": int(lag), "n": len(xs),
            "slope": float(slope), "intercept": float(icpt),
            "cmd_for_equal_wheels": float(-icpt / slope) if slope != 0 else None,
            "note": "trim 실측(-0.051, docs/1011.md §2)과 비교. 단, 명령 분포가 trim 근처에 몰려 있으면 외삽 오차 큼",
            "cmd_values_hist": {f"{v:+.2f}": int(c) for v, c in
                                zip(*np.unique(np.round(xs, 2), return_counts=True))},
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / "audit_per_run.json", "w") as f:
        json.dump(per_run, f, indent=2, ensure_ascii=False)
    plot_audit(out_dir, P, mask, corr, unit_pts, scale_pts, freeze_all, best_pair)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[done] {out_dir}")


def plot_audit(out_dir, P, mask, corr, unit_pts, scale_pts, freeze_all, best_pair):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    # (a) 동결 길이 분포
    ax = axs[0, 0]
    if freeze_all:
        ax.hist(freeze_all, bins=40, color="#4a6fa5")
    ax.set_title("GPS freeze episode length (moving ticks)")
    ax.set_xlabel("seconds")

    # (b) gyro z vs cmd (best lag)
    ax = axs[0, 1]
    r = corr.get("gyro_axis2")
    if r:
        lag = r[0]
        idx = [i for i in range(lag, len(mask)) if mask[i] and mask[i - lag] and np.isfinite(P["gyro"][i, 2])]
        ax.scatter([P["cmd"][i - lag] for i in idx], [P["gyro"][i, 2] for i in idx], s=3, alpha=0.3)
        ax.set_title(f"gyro z (tick mean) vs effective angular cmd, lag={lag}, r={r[1]:.2f}")
    ax.set_xlabel("effective angular cmd")

    # (c) best rpm pairing diff vs cmd
    ax = axs[0, 2]
    L, R = RPM_PAIRINGS[best_pair]
    r = corr.get(f"rpm_{best_pair}")
    if r:
        lag = r[0]
        diff = P["rpm"][:, list(L)].sum(1) - P["rpm"][:, list(R)].sum(1)
        idx = [i for i in range(lag, len(mask)) if mask[i] and mask[i - lag] and np.isfinite(diff[i])]
        ax.scatter([P["cmd"][i - lag] for i in idx], [diff[i] for i in idx], s=3, alpha=0.3, color="#c0504d")
        ax.set_title(f"rpm {best_pair} vs cmd, lag={lag}, r={r[1]:.2f}")
    ax.set_xlabel("effective angular cmd")

    # (d) turn windows: gyro integral vs GPS Δcourse
    ax = axs[1, 0]
    if unit_pts["gyro_z"]:
        dc, s = np.array(unit_pts["gyro_z"]).T
        ax.scatter(np.degrees(dc), s, s=10)
    ax.set_title("turn windows: ∫gyro_z dt vs GPS Δcourse")
    ax.set_xlabel("GPS Δcourse (deg)")

    # (e) turn windows: rpm diff integral vs GPS Δcourse
    ax = axs[1, 1]
    if unit_pts[best_pair]:
        dc, s = np.array(unit_pts[best_pair]).T
        ax.scatter(np.degrees(dc), s, s=10, color="#c0504d")
    ax.set_title(f"turn windows: ∫rpm {best_pair} dt vs GPS Δcourse")
    ax.set_xlabel("GPS Δcourse (deg)")

    # (f) rpm 회전수 vs GPS 거리
    ax = axs[1, 2]
    if scale_pts:
        revs, dist = np.array(scale_pts).T
        ax.scatter(revs, dist, s=10, color="#9bbb59")
        s = np.sum(revs * dist) / np.sum(revs ** 2)
        xx = np.linspace(0, revs.max(), 10)
        ax.plot(xx, s * xx, "k--", lw=1, label=f"{s:.3f} m/rev")
        ax.legend()
    ax.set_title("straight segments: wheel revs vs GPS distance")
    ax.set_xlabel("Σ rpm/60·dt (rev)")
    ax.set_ylabel("GPS distance (m)")

    for a in axs.flat:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "audit_overview.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["audit"], help="현재 Phase A(audit)만 구현")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else REPO_ROOT / "deployment/analysis/results" / \
        f"fusion_replay_{time.strftime('%Y%m%d')}"
    if args.phase == "audit":
        audit(out)


if __name__ == "__main__":
    main()
