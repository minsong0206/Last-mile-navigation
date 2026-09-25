"""
generate_heading_route_diagram.py

캡스톤 발표용 "Heading vs Route Bearing" 비교 다이어그램.

두 케이스 모두 실제 로그/코드 실행 결과 값을 그대로 사용 (임의 수치 없음):

- Case A (정렬된 경우): 2026-09-25 offline_smoke_test.py dry_run=True 실행
  (deployment/logs/deploy_20260925_165933.jsonl), tick_id=11.
  GPS-track heading -111.6° vs route bearing -132.4° → 차이 +20.8°
  (heading_route_diff_deg 필드 그대로)

- Case B (심한 불일치): 2026-09-18 실배포 중 angular=+0.3 포화가 관측된 순간
  (deployment/logs/deploy_20260918_181503.jsonl, ts=1789722955.50 —
  이전 세션에서 "0918 failure replay"로 이미 분석한 바로 그 순간).
  그날 실제 사용된 IMU 폴백 heading -192.0° vs 오늘 추가한
  LiveMapBuilder.route_bearing_rad()로 그 지점 기준 다시 계산한 route bearing
  -105.7° → 차이 -86.3°. (참고용으로 그 순간 GPS-track heading(-128.4°, 2026-09-18
  세션에서 사후 수정된 함수로 재계산한 값)을 썼다면 차이는 -22.6°까지 줄었을 것 —
  즉 "GPS-heading을 못 써서 IMU로 폴백했던 게 이 정도 오차의 핵심 원인이었다"는
  근거로 같이 표기.

실행: (frodobot conda env, 외부 연결 불필요 — 이미 계산된 값을 이 스크립트에 상수로
       박아둠. 원본 계산 과정은 위 주석 참고, 재검증하려면 해당 run의 JSONL을 다시 열면 됨)
    python3 docs/deployment_visualization/03_heading_route/generate_heading_route_diagram.py
출력: docs/deployment_visualization/03_heading_route/heading_route_diagram.png
"""
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent
FONT_DIR = Path("/usr/share/fonts/opentype/noto")
FONT_REG = FONT_DIR / "NotoSansCJK-Regular.ttc"
FONT_BOLD = FONT_DIR / "NotoSansCJK-Bold.ttc"


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REG), size)


# ── 실제 로그 값 (위 docstring 출처 그대로) ──
CASE_A = dict(
    title="Case A — 비교적 정렬됨",
    source="2026-09-25 offline_smoke_test.py (dry_run=True)\ndeploy_20260925_165933.jsonl, tick_id=11",
    heading_deg=-111.6, heading_label="Final Heading (GPS-track)",
    route_deg=-132.4, diff_deg=20.8,
)
CASE_B = dict(
    title="Case B — 심한 불일치 (0918 실배포 angular=+0.3 포화 순간)",
    source="2026-09-18 deploy_20260918_181503.jsonl, ts=1789722955.50\n"
           "route_bearing은 오늘(2026-09-25) 추가한 route_bearing_rad()로 재계산",
    heading_deg=-192.0, heading_label="Final Heading (IMU 폴백)",
    route_deg=-105.7, diff_deg=-86.3,
    extra="참고: 그 순간 GPS-track heading(-128.4°, 2026-09-18 세션에서 사후 수정된 함수로\n"
          "재계산)을 썼다면 차이는 -22.6°까지 줄었을 것 — IMU 폴백 자체가 오차의 핵심 원인이었음",
)

C_TEXT = (17, 24, 39)
C_SUB = (75, 85, 99)
C_HEADING = (37, 99, 235)   # 파랑 — 로봇 heading
C_ROUTE = (220, 38, 38)     # 빨강 — route bearing
C_DIFF = (194, 65, 12)      # 주황 — Δθ


def compass_deg_to_xy(cx, cy, r, deg):
    """East=0, North=+90 CCW 컨벤션(estimate_heading_from_track과 동일)을
    화면 좌표(위=북쪽, CCW가 반시계)로 변환."""
    rad = math.radians(deg)
    return cx + r * math.cos(rad), cy - r * math.sin(rad)


def draw_arrow(d, cx, cy, deg, r, color, width=6, label=None, label_font=None):
    x, y = compass_deg_to_xy(cx, cy, r, deg)
    d.line([(cx, cy), (x, y)], fill=color, width=width)
    ang = math.radians(deg)
    for da in (0.45, -0.45):
        hx = x - 22 * math.cos(ang - da)
        hy = y + 22 * math.sin(ang - da)
        d.line([(x, y), (hx, hy)], fill=color, width=width)
    if label:
        lx, ly = compass_deg_to_xy(cx, cy, r + 65, deg)
        bbox = d.textbbox((0, 0), label, font=label_font)
        d.text((lx - (bbox[2] - bbox[0]) / 2, ly - (bbox[3] - bbox[1]) / 2), label, font=label_font, fill=color)


def draw_diff_arc(d, cx, cy, r, deg1, deg2, color, width=4):
    a1, a2 = -deg1, -deg2  # PIL arc는 시계방향 각도(화면 좌표) 기준이라 부호 반전
    lo, hi = min(a1, a2), max(a1, a2)
    if hi - lo > 180:
        lo, hi = hi - 360, lo
    d.arc([cx - r, cy - r, cx + r, cy + r], lo, hi, fill=color, width=width)


def draw_case(d, cx, cy, case, F_TITLE, F_SUB, F_LABEL, F_DIFF):
    R = 260
    # 나침반 원 + 방위 눈금
    d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(209, 213, 219), width=2)
    for deg in range(0, 360, 30):
        x0, y0 = compass_deg_to_xy(cx, cy, R - 8, deg)
        x1, y1 = compass_deg_to_xy(cx, cy, R, deg)
        d.line([(x0, y0), (x1, y1)], fill=(209, 213, 219), width=2)
    for deg, lab in [(90, "N"), (0, "E"), (-90, "S"), (180, "W")]:
        x, y = compass_deg_to_xy(cx, cy, R + 22, deg)
        bbox = d.textbbox((0, 0), lab, font=F_SUB)
        d.text((x - (bbox[2] - bbox[0]) / 2, y - (bbox[3] - bbox[1]) / 2), lab, font=F_SUB, fill=(156, 163, 175))

    draw_diff_arc(d, cx, cy, R * 0.45, case["heading_deg"], case["route_deg"], C_DIFF, width=5)
    draw_arrow(d, cx, cy, case["route_deg"], R * 0.8, C_ROUTE, label="Route Bearing", label_font=F_LABEL)
    draw_arrow(d, cx, cy, case["heading_deg"], R * 0.8, C_HEADING, label=case["heading_label"], label_font=F_LABEL)
    d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=(17, 24, 39))

    # 제목/수치/출처
    bbox = d.textbbox((0, 0), case["title"], font=F_TITLE)
    d.text((cx - (bbox[2] - bbox[0]) / 2, cy - R - 90), case["title"], font=F_TITLE, fill=C_TEXT)

    diff_txt = f"Δθ (heading - route) = {case['diff_deg']:+.1f}°"
    bbox = d.textbbox((0, 0), diff_txt, font=F_DIFF)
    d.text((cx - (bbox[2] - bbox[0]) / 2, cy + R + 20), diff_txt, font=F_DIFF, fill=C_DIFF)

    detail = f"heading={case['heading_deg']:+.1f}°   route_bearing={case['route_deg']:+.1f}°"
    bbox = d.textbbox((0, 0), detail, font=F_SUB)
    d.text((cx - (bbox[2] - bbox[0]) / 2, cy + R + 60), detail, font=F_SUB, fill=C_SUB)

    y = cy + R + 100
    for line in case["source"].split("\n"):
        bbox = d.textbbox((0, 0), line, font=F_SUB)
        d.text((cx - (bbox[2] - bbox[0]) / 2, y), line, font=F_SUB, fill=(156, 163, 175))
        y += 26
    if "extra" in case:
        y += 8
        for line in case["extra"].split("\n"):
            bbox = d.textbbox((0, 0), line, font=F_SUB)
            d.text((cx - (bbox[2] - bbox[0]) / 2, y), line, font=F_SUB, fill=C_DIFF)
            y += 26


def main():
    W, H = 1900, 950
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)

    F_TITLE_MAIN = font(38, bold=True)
    F_SUB_MAIN = font(20)
    F_TITLE = font(24, bold=True)
    F_SUB = font(18)
    F_LABEL = font(19, bold=True)
    F_DIFF = font(22, bold=True)

    d.text((60, 30), "Heading vs Route Bearing — 왜 heading_route_diff_deg를 추가했는가", font=F_TITLE_MAIN, fill=C_TEXT)
    d.text((60, 80), "파란 화살표 = 로봇이 실제로 향한 방향(Final Heading), 빨간 화살표 = 경로를 따라가려면 가야 할 방향(Route Bearing)",
           font=F_SUB_MAIN, fill=C_SUB)

    draw_case(d, 480, 470, CASE_A, F_TITLE, F_SUB, F_LABEL, F_DIFF)
    draw_case(d, 1420, 470, CASE_B, F_TITLE, F_SUB, F_LABEL, F_DIFF)

    d.line([(950, 150), (950, 850)], fill=(229, 231, 235), width=2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "heading_route_diagram.png"
    img.save(out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
