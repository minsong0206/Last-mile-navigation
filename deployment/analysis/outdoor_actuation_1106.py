"""실외 작동(actuation) 특성 확정 — 2026-11-06 teleop + deploy LIVE 데이터.

목적: "모델 출력 → 로봇 실제 회전"이 실외에서 어떻게 되는지를 확정한다. 각 teleop run은
고정 조향 명령(angular_bias) 하나를 수 초간 유지하므로, 그동안의 자이로 z(=yaw rate,
deg/s, 2026-11-01 Phase B에서 단위 확정)의 평균이 그 명령에 대한 정상상태 회전속도다.

측정: 명령값별 평균 yaw rate → (1) 데드밴드(회전이 0인 최대 |명령|), (2) 게인(그 위의
deg/s per cmd), (3) 좌우 대칭. 실내값(데드밴드 0.05, 게인 1.66 rad/s/cmd ≈ 95 deg/s/cmd)과 비교.

자이로는 버스트로 와서 시간 커버리지가 낮지만(~17%), 명령이 일정하게 유지되는 구간의
표본 평균은 정상상태 yaw rate의 불편추정이다(어느 17%인지와 무관). 응답 지연(3~5틱)을
감안해 각 run의 앞 구간(워밍업)은 버린다.

실행: python3 deployment/analysis/outdoor_actuation_1106.py
"""
import json
import glob
import math
import statistics as st
from collections import defaultdict

RAD2DEG = 180.0 / math.pi
INDOOR_GAIN_DEG_S_PER_CMD = 95.0   # 1.66 rad/s/cmd
INDOOR_DEADBAND = 0.05

WARMUP_FRAC = 0.35   # 각 run 앞 35% 틱 버림(응답 지연/가속 전이 제거)


def run_mean_yawrate(path):
    """teleop run 하나 → (cmd, 정상상태 yaw_rate_deg_s, n_samples, mean_lrpm, mean_rrpm)."""
    lines = [json.loads(l) for l in open(path)]
    rs = lines[0]
    cmd = rs.get("angular_bias")
    steps = [l for l in lines if l.get("type") == "step"]
    if len(steps) < 5:
        return None
    steady = steps[int(len(steps) * WARMUP_FRAC):]
    gz = []
    lrpm = []
    rrpm = []
    for s in steady:
        for g in (s.get("gyros") or []):
            if len(g) >= 3:
                gz.append(g[2])              # 자이로 z = yaw rate (deg/s), +=좌
        for r in (s.get("rpms") or []):
            if len(r) >= 4:
                lrpm.append((r[0] + r[2]) / 2.0)   # ch0=ch2 왼쪽
                rrpm.append((r[1] + r[3]) / 2.0)   # ch1=ch3 오른쪽
    if len(gz) < 3:
        return None
    return (cmd, st.mean(gz), len(gz),
            st.mean(lrpm) if lrpm else None, st.mean(rrpm) if rrpm else None)


def main():
    # ── teleop: 명령값별 집계 ──
    by_cmd = defaultdict(list)
    per_run = []
    for f in sorted(glob.glob("deployment/logs/1106_test/teleop_test_*.jsonl")):
        r = run_mean_yawrate(f)
        if r is None:
            continue
        cmd, yaw, n, lr, rr = r
        by_cmd[cmd].append(yaw)
        per_run.append((cmd, yaw, n, lr, rr, f.split("/")[-1]))

    print("=" * 72)
    print("실외 작동 특성 — teleop (명령 일정 유지, 자이로 z = yaw rate deg/s, +=좌)")
    print("=" * 72)
    print(f"{'cmd':>8} {'n_run':>5} {'yaw_rate(deg/s)':>16} {'std':>7} {'좌우rpm차(평균)':>14}")
    cmds = sorted(by_cmd)
    agg = {}
    for c in cmds:
        ys = by_cmd[c]
        m = st.mean(ys)
        sd = st.pstdev(ys) if len(ys) > 1 else 0.0
        agg[c] = m
        # 같은 cmd run들의 좌우 rpm차 평균
        lrrr = [(lr, rr) for cc, yy, nn, lr, rr, ff in per_run if cc == c and lr is not None]
        rpmdiff = st.mean([lr - rr for lr, rr in lrrr]) if lrrr else float("nan")
        print(f"{c:+8.3f} {len(ys):5d} {m:16.1f} {sd:7.1f} {rpmdiff:14.2f}")

    # ── 데드밴드 / 게인 추정 (우회전 쪽: cmd<0) ──
    # 방법: "회전하는" 점들(|yaw|가 노이즈보다 확실히 큰 큰 명령)에 affine 최소제곱
    # (y = G·cmd + b)을 피팅한다. 원점통과를 가정하지 않으므로 데드밴드 오프셋을
    # 제대로 잡는다. x절편(y=0 되는 cmd)이 실효 데드밴드, 기울기 G가 게인.
    neg = [(c, agg[c]) for c in cmds if c < 0]
    print()
    print("─" * 72)

    def affine_fit(pts):
        n = len(pts)
        mx = sum(c for c, y in pts) / n
        my = sum(y for c, y in pts) / n
        sxx = sum((c - mx) ** 2 for c, y in pts)
        sxy = sum((c - mx) * (y - my) for c, y in pts)
        if sxx == 0:
            return None
        G = sxy / sxx                 # deg/s per cmd
        b = my - G * mx               # deg/s intercept
        x0 = -b / G if G != 0 else float("nan")  # x절편 = 실효 데드밴드(부호 포함)
        return G, b, x0

    # 회전 점: 명령 크기가 충분히 큰 것(|cmd|>=0.07)만 — 데드밴드 근처 soft knee 제외
    turn_pts = [(c, y) for c, y in neg if abs(c) >= 0.07 - 1e-9]
    fit = affine_fit(turn_pts) if len(turn_pts) >= 2 else None
    if fit:
        G, b, x0 = fit
        db = abs(x0)
        print(f"[우회전(cmd<0)] affine 피팅 점 {len(turn_pts)}개 (|cmd|>=0.07):")
        print(f"    게인 G = {G:.1f} deg/s per cmd = {G/RAD2DEG:.3f} rad/s per cmd")
        print(f"    실효 데드밴드(x절편) ≈ {db:.3f}")
        print(f"  (실내: 데드밴드 {INDOOR_DEADBAND}, 게인 {INDOOR_GAIN_DEG_S_PER_CMD:.0f} deg/s per cmd "
              f"= {INDOOR_GAIN_DEG_S_PER_CMD/RAD2DEG:.2f} rad/s per cmd)")
        print(f"  → 실외 게인 / 실내 게인 = {G/INDOOR_GAIN_DEG_S_PER_CMD:.2f}배")
    else:
        G = db = float("nan")
        print("[우회전] 회전 점이 부족해 피팅 불가.")

    # 좌회전 표본
    pos = [(c, agg[c]) for c in cmds if c > 0]
    if pos:
        print(f"[좌회전(cmd>0)] 표본: " + ", ".join(f"cmd{c:+.2f}→{y:+.1f}deg/s" for c, y in pos))

    # ── 보정식 파라미터 도출 (desired yaw rate → 보낼 command 역변환) ──
    if fit and not math.isnan(G) and G != 0:
        G_rad = G / RAD2DEG           # rad/s per cmd
        print()
        print("─" * 72)
        print("조향 복원 보정 파라미터 (모델 desired ω[rad/s] → 보낼 angular cmd)")
        print(f"  cmd = sign(ω)·DB + ω / Gout   (clamp ±0.30)")
        print(f"    DB   = {db:.3f}           (실효 데드밴드, 역변환 오프셋)")
        print(f"    Gout = {G_rad:.3f} rad/s/cmd  (실외 게인)")
        print(f"    입력 데드존: |ω| < EPS 이면 cmd=0 (모델이 진짜 직진일 때 ±DB 안 쏘게)")
        print(f"    권장 EPS ≈ 0.010 rad/s (≈0.6 deg/s)")
        print("  검증 (desired → 보낼 cmd → 이 표가 예측하는 실제 yaw):")
        for omega in (-0.02, -0.05, -0.10, -0.15):
            sent = math.copysign(db, omega) + omega / G_rad
            sent_c = max(-0.30, min(0.30, sent))
            pred = G_rad * (abs(sent_c) - db) if abs(sent_c) > db else 0.0
            pred = math.copysign(pred, sent_c)
            print(f"    ω={omega:+.2f} rad/s → cmd {sent_c:+.3f} → 실제≈{pred:+.3f} rad/s "
                  f"({pred*RAD2DEG:+.1f} deg/s)")

    # ── deploy LIVE 교차검증 (gyro_rate_deg_s 필드가 있으면) ──
    print()
    print("─" * 72)
    print("deploy LIVE 교차검증 (전송 angular vs 로그의 gyro_rate_deg_s)")
    pairs = []
    for f in sorted(glob.glob("deployment/logs/1011/deploy_20261106_*.jsonl")):
        lines = [json.loads(l) for l in open(f)]
        live = [l for l in lines if l.get("type") == "step" and l.get("control_stage") == "LIVE"]
        for s in live:
            gr = s.get("gyro_rate_deg_s")
            ang = s.get("angular")
            if gr is not None and ang is not None and s.get("gyro_moving"):
                pairs.append((ang, gr))
    if pairs:
        sent = [a for a, g in pairs]
        in_db = sum(1 for a in sent if abs(a) <= INDOOR_DEADBAND)
        print(f"  gyro_moving 틱 {len(pairs)}개 중 전송 |angular|≤0.05(데드밴드) = {in_db} "
              f"({in_db/len(pairs)*100:.0f}%)")
        turn = [(a, g) for a, g in pairs if abs(a) > 0.05]
        if turn:
            gain_d = sum(g * a for a, g in turn) / sum(a * a for a, g in turn)
            print(f"  데드밴드 밖 {len(turn)}틱 게인 ≈ {gain_d:.1f} deg/s per cmd "
                  f"= {gain_d/RAD2DEG:.3f} rad/s per cmd")
        else:
            print("  데드밴드 밖(|angular|>0.05) 전송 틱이 없음 — 모델 명령이 전부 데드밴드 안.")
    else:
        print("  gyro_rate_deg_s 있는 LIVE 틱 없음.")

    print()
    print("=" * 72)
    print("결론 재료: 위 데드밴드/게인이 실내값과 얼마나 다른지 + 모델 명령이 데드밴드에 얼마나 묶이는지")


if __name__ == "__main__":
    main()
