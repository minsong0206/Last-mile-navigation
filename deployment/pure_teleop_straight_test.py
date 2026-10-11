"""순수 teleop 직진 테스트 (2026-10-11).

좌편향이 모델/override/cross-track 로직과 완전히 무관한 하드웨어(바퀴 등) 쪽
편향인지 확인하기 위한 최소 스크립트 — omnivla_edge_deploy.py의 어떤 로직도
타지 않고 고정 angular(기본 0.0, --angular_bias로 트림값 탐색 가능)만 전송한다.
시작/종료 GPS로 실제 이동 방향을 계산해서 출력 — 로봇이 바라보던 방향과 다르게
(특히 왼쪽으로) 틀어져 있으면 하드웨어 편향 가설이 뒷받침된다.

2026-10-11 추가(2차, 실내 반복 테스트에서 angular=0.0인데도 매번 좌측으로 드리프트
실측 확인됨 — 컴퍼스는 매 run 다른 패턴으로 흔들려 신뢰 불가했지만 사용자의 직접
관찰은 3회 반복 일관됨): --angular_bias로 고정 보정값(예: -0.05, 우측 보정)을 넣어
실제로 상쇄되는 트림값을 실측 탐색할 수 있게 함.

매 tick JSONL 로그도 남긴다(--log_dir, 기본 deployment/logs/1011) — /data 텔레메트리
(위경도/컴퍼스 orientation/fix_quality)를 매 tick 함께 기록해서, GPS가 불안정한
실내에서도 최소한 컴퍼스 orientation의 변화를 나중에 분석할 수 있게 함. 텔레메트리
조회 실패는 로그에만 기록하고 테스트 자체는 계속 진행(제어 명령 전송이 최우선)."""
import argparse
import json
import math
import time
from pathlib import Path

import requests

FRODOBOT_BASE = "http://127.0.0.1:8000"
LAT_M = 111320.0
REPO_ROOT = Path(__file__).resolve().parent.parent


def get_data():
    r = requests.get(f"{FRODOBOT_BASE}/data", timeout=5.0)
    r.raise_for_status()
    return r.json()


def get_gps():
    d = get_data()
    return d["latitude"], d["longitude"]


def send_control(linear, angular):
    r = requests.post(f"{FRODOBOT_BASE}/control",
                       json={"command": {"linear": linear, "angular": angular}}, timeout=5.0)
    return r.status_code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--linear", type=float, default=0.2,
                   help="고정 선속도 m/s (기본 0.2, MAX_V=0.3보다 낮춰서 안전하게).")
    p.add_argument("--angular_bias", type=float, default=0.0,
                   help="매 tick 고정으로 더하는 angular 트림 (기본 0.0=순수 직진 테스트). "
                        "양수=좌회전/음수=우회전 관례(프로젝트 전체 동일) — 실측으로 확인된 "
                        "좌측 드리프트를 상쇄하려면 음수 값(예: -0.05)을 줄 것. 이 값을 주면 "
                        "더는 '순수' 테스트가 아니라 '하드웨어 트림 보정값 탐색' 테스트가 됨 — "
                        "로그의 run_start.angular_bias로 항상 구분 가능.")
    p.add_argument("--enable_angular_dither", action="store_true",
                   help="angular 양자화 디더링(델타-시그마) 활성화. 로봇이 angular를 "
                        "--angular_quantum 단위로 0 쪽으로 절삭하는 것이 실측으로 확정됨 "
                        "(-0.05/-0.055/-0.058/-0.059가 전부 동일하게 동작, -0.06만 달라짐). "
                        "그래서 두 양자화 단계 사이의 트림값은 그냥 보내면 사라지고, 하드웨어 "
                        "편향의 진짜 균형점을 못 찾는다. 이 옵션을 켜면 양자화 잔차를 다음 tick에 "
                        "넘겨 인접 두 단계를 섞어 보내서 시간평균이 요청값에 수렴함 — "
                        "0.001 해상도로 하드웨어 트림을 실측할 수 있게 됨.")
    p.add_argument("--angular_quantum", type=float, default=0.01,
                   help="위 디더링이 가정하는 로봇 측 angular 양자화 단위 (기본 0.01, 실측값).")
    p.add_argument("--duration", type=float, default=15.0, help="주행 시간(초), 기본 15초")
    p.add_argument("--rate_hz", type=float, default=3.0, help="명령 전송 주기 (기본 3Hz, 배포 루프와 동일)")
    p.add_argument("--skip_gps", action="store_true",
                   help="실내 테스트용 — GPS fix가 없거나 불안정한 곳에서는 시작/종료 GPS 비교가 "
                        "무의미하므로 요약 비교만 생략. 매 tick 로그에는 텔레메트리를 여전히 "
                        "(가능하면) 기록함. 바닥에 직선을 긋고 눈으로 드리프트를 확인하는 방식으로 "
                        "판단할 것.")
    p.add_argument("--log_dir", type=str, default=str(REPO_ROOT / "deployment" / "logs" / "1011"),
                   help="JSONL 로그 저장 폴더 (기본 deployment/logs/1011, 배포 스크립트와 동일 날짜 폴더)")
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"teleop_test_{run_id}.jsonl"
    log_f = open(log_path, "a")

    def log_jsonl(record):
        log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_f.flush()

    log_jsonl({"type": "run_start", "ts": time.time(), "run_id": run_id,
               "linear": args.linear, "angular_bias": args.angular_bias,
               "duration": args.duration, "rate_hz": args.rate_hz,
               "skip_gps": args.skip_gps,
               "enable_angular_dither": args.enable_angular_dither,
               "angular_quantum": args.angular_quantum})
    print(f"[teleop_test] 로그 저장 경로: {log_path}")

    # 2026-10-11 추가: SDK 서버의 헤드리스 브라우저(pyppeteer)는 첫 요청에서 Chrome 실행+
    # /sdk 접속+RTM join까지 해서 수 초~십수 초가 걸림(omnivla_edge_deploy.py::run()과 동일
    # 이슈) — 그 콜드스타트 중에 5초 타임아웃의 /control을 먼저 보내면 ReadTimeout으로
    # 죽음(실측: 15초 주행 시작 직후 crash, step 로그 한 줄도 못 남김). 느긋한 타임아웃으로
    # 먼저 한 번 깨워둠.
    print("[teleop_test] SDK 서버 워밍업 중 (헤드리스 브라우저 초기화 대기)...")
    try:
        requests.get(f"{FRODOBOT_BASE}/data", timeout=30.0)
    except Exception as e:
        print(f"[teleop_test] ⚠ 워밍업 요청 실패 — SDK 서버가 안 떠 있을 수 있음: {e!r}")
        log_jsonl({"type": "event", "ts": time.time(), "event": "warmup_failed", "error": repr(e)})
        log_f.close()
        return

    if args.skip_gps:
        print("[teleop_test] --skip_gps: GPS 비교 생략 (실내 테스트) — 바닥 기준선 대비 눈으로 "
              "드리프트를 확인하세요.")
        lat0 = lon0 = None
    else:
        print("[teleop_test] 시작 GPS 확인 중...")
        lat0, lon0 = get_gps()
        print(f"[teleop_test] 시작: lat={lat0:.7f} lon={lon0:.7f}")
    print(f"[teleop_test] linear={args.linear:+.2f} angular={args.angular_bias:+.4f} 고정으로 "
          f"{args.duration:.0f}초간 주행 ({args.rate_hz:.1f}Hz) — Ctrl+C로 언제든 즉시 정지")

    period_s = 1.0 / args.rate_hz
    t_start = time.time()
    tick_id = 0
    dither_accum = 0.0
    sent_history = []
    try:
        while time.time() - t_start < args.duration:
            tick_t0 = time.time()
            # 델타-시그마 디더링: 양자화 잔차를 다음 tick에 넘겨 인접 두 단계를 섞어
            # 보내면 시간평균이 요청값에 수렴한다 — 안 하면 요청값이 절삭돼 사라져서
            # 0.001 단위 트림 탐색이 아예 불가능함(--enable_angular_dither 설명 참고).
            if args.enable_angular_dither and args.angular_quantum > 0:
                q = args.angular_quantum
                # round()로 부동소수 표현오차 제거 후 truncate — 안 하면 -0.06이
                # -0.059999...로 저장돼 한 단계 손실됨.
                steps = math.trunc(round((args.angular_bias + dither_accum) / q, 6))
                angular_sent = steps * q
                dither_accum = max(-2.0 * q, min(2.0 * q,
                                                 dither_accum + (args.angular_bias - angular_sent)))
            else:
                angular_sent = args.angular_bias
            sent_history.append(angular_sent)
            status = send_control(args.linear, angular_sent)
            elapsed = time.time() - t_start
            record = {"type": "step", "ts": time.time(), "tick_id": tick_id,
                      "elapsed_s": elapsed, "linear": args.linear,
                      "angular": angular_sent, "angular_desired": args.angular_bias,
                      "angular_dither_accum": dither_accum,
                      "control_status": status}
            try:
                data = get_data()
                record["lat"] = data.get("latitude")
                record["lon"] = data.get("longitude")
                record["orientation_deg_raw"] = data.get("orientation")
                record["fix_quality"] = data.get("fix_quality")
            except Exception as e:
                record["telemetry_error"] = repr(e)
            log_jsonl(record)
            print(f"  t={elapsed:5.1f}s linear={args.linear:+.3f} angular={angular_sent:+.3f}"
                  f"(목표 {args.angular_bias:+.4f}) [/control {status}] "
                  f"orientation={record.get('orientation_deg_raw')}")
            tick_id += 1
            sleep_s = period_s - (time.time() - tick_t0)
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\n[teleop_test] 중단 요청됨")
        log_jsonl({"type": "event", "ts": time.time(), "event": "keyboard_interrupt"})
    finally:
        print("[teleop_test] 정지 명령 전송")
        try:
            send_control(0.0, 0.0)
        except Exception as e:
            print(f"[teleop_test] ⚠ 정지 명령 전송 실패 (서버 연결 확인 필요): {e!r}")

    # 디더링이 실제로 요청값을 실현했는지 검증 — 평균이 목표와 다르면 측정 자체를
    # 신뢰할 수 없으므로 여기서 바로 드러나게 한다.
    if sent_history:
        avg = sum(sent_history) / len(sent_history)
        steps_used = sorted(set(round(v, 4) for v in sent_history))
        print(f"[teleop_test] 전송된 angular: 평균={avg:+.5f} (목표 {args.angular_bias:+.4f}, "
              f"오차 {avg - args.angular_bias:+.5f}), 사용된 단계={steps_used}")
        if args.enable_angular_dither and abs(avg - args.angular_bias) > 0.002:
            print("[teleop_test] ⚠ 평균이 목표와 0.002 이상 차이남 — tick 수가 너무 적을 수 있음"
                  "(duration을 늘리면 수렴). 이 run의 트림 판정은 신뢰도가 낮음.")
        log_jsonl({"type": "dither_summary", "ts": time.time(), "n_ticks": len(sent_history),
                   "angular_desired": args.angular_bias, "angular_sent_avg": avg,
                   "steps_used": steps_used})

    if args.skip_gps:
        if args.angular_bias != 0.0:
            print(f"[teleop_test] (--skip_gps, angular_bias={args.angular_bias:+.4f}) 바닥 기준선 "
                  f"대비 이번엔 거의 똑바로 갔는지 확인하세요 — 거의 직진이면 이 트림값이 하드웨어 "
                  f"편향을 상쇄하는 근삿값, 여전히 치우치면 트림값을 더 조정해서 재시도.")
        else:
            print("[teleop_test] (--skip_gps) 바닥 기준선 대비 로봇이 왼쪽/오른쪽 중 어느 쪽으로 "
                  "벗어났는지 눈으로 확인한 결과를 기록해두세요 — 왼쪽으로 벗어났으면 하드웨어(바퀴) "
                  "좌편향 가설이 뒷받침됩니다.")
        log_jsonl({"type": "run_end", "ts": time.time(), "reason": "duration_or_interrupt",
                   "gps_comparison": None})
        log_f.close()
        print(f"[teleop_test] 로그 저장 완료: {log_path}")
        return

    print("[teleop_test] 종료 GPS 확인 중...")
    lat1, lon1 = get_gps()
    print(f"[teleop_test] 종료: lat={lat1:.7f} lon={lon1:.7f}")

    dlat_m = (lat1 - lat0) * LAT_M
    dlon_m = (lon1 - lon0) * LAT_M * math.cos(math.radians(lat0))
    dist_m = math.hypot(dlat_m, dlon_m)
    heading_deg = None
    if dist_m > 1e-3:
        heading_deg = math.degrees(math.atan2(dlat_m, dlon_m))
        print(f"[teleop_test] 실제 이동: {dist_m:.2f}m, 방향(East=0/North=90/CCW)={heading_deg:+.1f}°")
    else:
        print(f"[teleop_test] 실제 이동이 거의 없음 (GPS 변화 미미, dist={dist_m:.2f}m) — GPS 샘플링/"
              f"정확도가 이 테스트를 판단하기엔 부족할 수 있음, duration을 늘려서 재시도 권장")
    print("[teleop_test] -> 이 방향이 로봇이 테스트 시작 시 실제로 바라보던 방향과 다르면(특히 "
          "왼쪽으로 틀어져 있으면) 하드웨어(바퀴) 좌편향 가설이 뒷받침됩니다. 직접 비교해주세요.")

    log_jsonl({"type": "run_end", "ts": time.time(), "reason": "duration_or_interrupt",
               "gps_comparison": {"lat0": lat0, "lon0": lon0, "lat1": lat1, "lon1": lon1,
                                   "dist_m": dist_m, "heading_deg": heading_deg}})
    log_f.close()
    print(f"[teleop_test] 로그 저장 완료: {log_path}")


if __name__ == "__main__":
    main()
