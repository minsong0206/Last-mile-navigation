"""
generate_pipeline_diagram.py

캡스톤 발표용 "Deployment Pipeline" 아키텍처 다이어그램 생성 스크립트.
실제 FrodoBot/SDK 연결 없이, 순수 그림만 그린다 (수치/로그 데이터 사용 안 함 —
02/03/04와 달리 이 다이어그램은 "구조"를 보여주는 용도라 실측 데이터가 필요 없음).

실행: (frodobot conda env, 외부 연결 불필요)
    python3 docs/deployment_visualization/01_pipeline/generate_pipeline_diagram.py
출력: docs/deployment_visualization/01_pipeline/pipeline_diagram.png
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent
FONT_DIR = Path("/usr/share/fonts/opentype/noto")
FONT_REG = FONT_DIR / "NotoSansCJK-Regular.ttc"
FONT_BOLD = FONT_DIR / "NotoSansCJK-Bold.ttc"

W, H = 2200, 2080
BG = (255, 255, 255)

# 색상 팔레트
C_SENSOR = (219, 234, 254)     # 연한 파랑 — 센서 입력
C_SENSOR_B = (37, 99, 235)
C_PROC = (229, 231, 235)       # 연한 회색 — 기존 처리 단계
C_PROC_B = (75, 85, 99)
C_MODEL = (237, 233, 254)      # 연한 보라 — 모델
C_MODEL_B = (109, 40, 217)
C_CTRL = (209, 250, 229)       # 연한 초록 — 제어/출력
C_CTRL_B = (5, 150, 105)
C_NEW = (255, 237, 213)        # 연한 주황 — 이번에 추가한 것
C_NEW_B = (194, 65, 12)
C_ROBOT = (254, 226, 226)
C_ROBOT_B = (185, 28, 28)
C_TEXT = (17, 24, 39)
C_TEXT_SUB = (75, 85, 99)


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REG), size)


F_TITLE = font(40, bold=True)
F_SUB = font(22)
F_LABEL = font(22, bold=True)
F_SMALL = font(17)
F_LEGEND = font(19)


def rrect(d, xy, fill, outline, width=2, radius=14, dashed=False):
    x0, y0, x1, y1 = xy
    if not dashed:
        d.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
        return
    d.rounded_rectangle(xy, radius=radius, fill=fill)
    # 점선 테두리 (rounded_rectangle은 dash 옵션이 없어서 짧은 선분을 이어붙여 흉내)
    perim = []
    n = 60
    import math
    cx0, cy0, cx1, cy1 = x0, y0, x1, y1
    pts = [(cx0, cy0), (cx1, cy0), (cx1, cy1), (cx0, cy1), (cx0, cy0)]
    for i in range(len(pts) - 1):
        (ax, ay), (bx, by) = pts[i], pts[i + 1]
        length = math.hypot(bx - ax, by - ay)
        steps = max(int(length // 14), 1)
        for s in range(steps):
            t0 = s / steps
            t1 = (s + 0.55) / steps
            p0 = (ax + (bx - ax) * t0, ay + (by - ay) * t0)
            p1 = (ax + (bx - ax) * t1, ay + (by - ay) * t1)
            d.line([p0, p1], fill=outline, width=width)


def text_center(d, cx, cy, lines, fnt, fill, line_gap=6):
    if isinstance(lines, str):
        lines = [lines]
    heights = []
    widths = []
    for ln in lines:
        bbox = d.textbbox((0, 0), ln, font=fnt)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    total_h = sum(heights) + line_gap * (len(lines) - 1)
    y = cy - total_h / 2
    for ln, wd, ht in zip(lines, widths, heights):
        d.text((cx - wd / 2, y), ln, font=fnt, fill=fill)
        y += ht + line_gap


def box(d, cx, cy, w, h, title, sub=None, fill=C_PROC, border=C_PROC_B, dashed=False, title_font=F_LABEL):
    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    rrect(d, (x0, y0, x1, y1), fill, border, width=3, dashed=dashed)
    if sub:
        text_center(d, cx, cy - 10, title, title_font, C_TEXT)
        text_center(d, cx, cy + h / 2 - 22, sub, F_SMALL, C_TEXT_SUB)
    else:
        text_center(d, cx, cy, title, title_font, C_TEXT)
    return (x0, y0, x1, y1)


def arrow(d, p0, p1, color=(55, 65, 81), width=3, dashed=False, label=None, label_font=F_SMALL):
    import math
    if dashed:
        length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        steps = max(int(length // 12), 1)
        for s in range(steps):
            t0 = s / steps
            t1 = (s + 0.5) / steps
            a = (p0[0] + (p1[0] - p0[0]) * t0, p0[1] + (p1[1] - p0[1]) * t0)
            b = (p0[0] + (p1[0] - p0[0]) * t1, p0[1] + (p1[1] - p0[1]) * t1)
            d.line([a, b], fill=color, width=width)
    else:
        d.line([p0, p1], fill=color, width=width)
    # 화살촉
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    sz = 11
    for da in (0.5, -0.5):
        a2 = ang + math.pi - da
        d.line([p1, (p1[0] + sz * math.cos(a2), p1[1] + sz * math.sin(a2))], fill=color, width=width)
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        d.text((mx + 8, my - 10), label, font=label_font, fill=color)


def main():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # ── 제목 ──
    d.text((60, 30), "OmniVLA-Edge FrodoBot Mini Deployment Pipeline", font=F_TITLE, fill=C_TEXT)
    d.text((60, 82), "기존: Sensor → Model → Control → Robot   →   현재: Sensor → Localization/Map → Model → Control"
                     " (+ Debug Panel / Dry-run Gate / Replay Logger)",
           font=F_SUB, fill=C_TEXT_SUB)

    # ── 범례 ──
    lx, ly = 60, 130
    for i, (label, fill, border) in enumerate([
        ("센서 입력", C_SENSOR, C_SENSOR_B),
        ("기존 처리 단계", C_PROC, C_PROC_B),
        ("모델", C_MODEL, C_MODEL_B),
        ("제어/출력", C_CTRL, C_CTRL_B),
        ("이번에 추가한 구조", C_NEW, C_NEW_B),
    ]):
        bx = lx + i * 260
        d.rounded_rectangle((bx, ly, bx + 26, ly + 26), radius=5, fill=fill, outline=border, width=2)
        d.text((bx + 34, ly + 3), label, font=F_LEGEND, fill=C_TEXT)

    top = 210
    # ══════════════════ 메인 파이프라인 (좌측, 위→아래) ══════════════════
    cx_main = 620

    # Row 1: 센서
    y1 = top + 40
    b_cam = box(d, 260, y1, 220, 90, "Camera", fill=C_SENSOR, border=C_SENSOR_B)
    b_gps = box(d, 620, y1, 220, 90, "GPS", fill=C_SENSOR, border=C_SENSOR_B)
    b_imu = box(d, 980, y1, 220, 90, "IMU", fill=C_SENSOR, border=C_SENSOR_B)

    # Row 2
    y2 = y1 + 180
    b_fb = box(d, 260, y2, 240, 100, "Frame Buffer", "최근 6프레임 (0.3초 간격)", fill=C_PROC, border=C_PROC_B)
    b_gh = box(d, 560, y2, 220, 100, "GPS Heading", "GPS 궤적 기반 추정", fill=C_PROC, border=C_PROC_B)
    b_ih = box(d, 980, y2, 220, 100, "IMU Heading", "컴퍼스 변환", fill=C_PROC, border=C_PROC_B)
    b_route = box(d, 1320, y2, 240, 100, "OSRM Route", "출발→목표 1회 계산·캐싱", fill=C_PROC, border=C_PROC_B)

    arrow(d, (260, y1 + 45), (260, y2 - 50))
    arrow(d, (600, y1 + 45), (570, y2 - 50))
    arrow(d, (940, y1 + 45), (990, y2 - 50))
    arrow(d, (660, y1 + 45), (1300, y2 - 50))  # GPS → OSRM Route

    # Row 3: Final Heading
    y3 = y2 + 170
    b_fh = box(d, 760, y3, 260, 100, "Final Heading", "GPS/IMU 선택 + EMA 스무딩", fill=C_PROC, border=C_PROC_B)
    arrow(d, (560, y2 + 50), (700, y3 - 50), label="GPS 이동량 충분")
    arrow(d, (980, y2 + 50), (830, y3 - 50), label="폴백/관성 유지")

    # Row 4: Heading-up Map
    y4 = y3 + 170
    b_map = box(d, 900, y4, 300, 110, "Heading-up Map", "Final Heading + Route → ego-centric 회전", fill=C_PROC, border=C_PROC_B)
    arrow(d, (760, y3 + 50), (830, y4 - 55))
    arrow(d, (1320, y2 + 50), (970, y4 - 55))

    # Row 5: OmniVLA
    y5 = y4 + 180
    b_model = box(d, 620, y5, 340, 120, "OmniVLA-Edge", "카메라 6프레임 + Heading-up Map", fill=C_MODEL, border=C_MODEL_B, title_font=font(26, bold=True))
    # Frame Buffer -> OmniVLA: 왼쪽 여백을 따라 곧게 내려와서 모델 박스 왼쪽으로 진입
    # (중간의 Final Heading/Heading-up Map 텍스트와 안 겹치게 x=140 고정 경로 사용)
    d.line([(260, y2 + 50), (140, y2 + 50)], fill=(55, 65, 81), width=3)
    d.line([(140, y2 + 50), (140, y5)], fill=(55, 65, 81), width=3)
    arrow(d, (140, y5), (450, y5))
    arrow(d, (900, y4 + 55), (760, y5 - 60))

    # Row 6: waypoints -> target -> controller -> command
    y6 = y5 + 190
    b_pred = box(d, 260, y6, 240, 100, "Predicted\nWaypoints", "8-step 궤적", fill=C_MODEL, border=C_MODEL_B)
    b_target = box(d, 560, y6, 220, 100, "Target Waypoint", "target_step 선택", fill=C_PROC, border=C_PROC_B)
    b_ctrl = box(d, 860, y6, 220, 100, "Controller", "waypoint → linear/angular", fill=C_CTRL, border=C_CTRL_B)
    b_calc = box(d, 1160, y6, 260, 100, "Calculated\nlinear/angular", "안전 클리핑 적용", fill=C_CTRL, border=C_CTRL_B)
    arrow(d, (620, y5 + 60), (400, y6 - 100), )
    d.line([(400, y6 - 100), (260, y6 - 50)], fill=(55, 65, 81), width=3)
    arrow(d, (380, y6), (450, y6))
    arrow(d, (670, y6), (750, y6))
    arrow(d, (970, y6), (1030, y6))

    # Row 7: Dry-run Gate (신규)
    y7 = y6 + 220
    b_gate = box(d, 1160, y7, 380, 150, "Dry-run Gate", "LIVE: calculated 그대로 전송\nDRY RUN: 항상 (0,0) 전송",
                 fill=C_NEW, border=C_NEW_B, dashed=True)
    arrow(d, (1160, y6 + 50), (1160, y7 - 75))

    # Row 8: Robot
    y8 = y7 + 240
    b_robot = box(d, 1160, y8, 260, 110, "FrodoBot Mini", "/control (RTM)", fill=C_ROBOT, border=C_ROBOT_B)
    arrow(d, (1060, y7 + 75), (1100, y8 - 55), label="LIVE")
    arrow(d, (1260, y7 + 75), (1220, y8 - 55), label="DRY RUN=(0,0)")

    # ══════════════════ 우측: 신규 구조 (Replay Logger) ══════════════════
    rx = 1800
    rl_top, rl_h = 330, 780
    d.rounded_rectangle((rx - 260, rl_top, rx + 260, rl_top + rl_h), radius=20,
                         outline=C_NEW_B, width=3)
    for yy in range(rl_top, rl_top + rl_h, 14):
        d.line([(rx - 260, yy), (rx - 260, min(yy + 7, rl_top + rl_h))], fill=C_NEW_B, width=3)
        d.line([(rx + 260, yy), (rx + 260, min(yy + 7, rl_top + rl_h))], fill=C_NEW_B, width=3)
    text_center(d, rx, rl_top + 34, "Replay Logger (신규)", font(24, bold=True), C_NEW_B)
    text_center(d, rx, rl_top + 66, "같은 tick_id로 실제 inference 입력을 그대로 저장", F_SMALL, C_TEXT_SUB)

    taps = [
        ("Camera Context\n(6 frames)", b_fb),
        ("Model-input Map", b_map),
        ("Prediction\n(waypoints/target)", b_pred),
        ("Heading & Route\n(final heading, bearing)", b_fh),
        ("Control\n(linear/angular)", b_calc),
    ]
    ty = rl_top + 120
    for label, src_box in taps:
        box(d, rx, ty, 420, 90, label.split("\n")[0],
            label.split("\n")[1] if "\n" in label else None,
            fill=(255, 247, 237), border=C_NEW_B, dashed=False, title_font=F_LABEL)
        sx1, sy1, sx2, sy2 = src_box
        arrow(d, (sx2, (sy1 + sy2) / 2), (rx - 210, ty), color=C_NEW_B, dashed=True, width=2)
        ty += 128

    text_center(d, rx, rl_top + rl_h - 30, "→ tick_id 하나로 그 순간 model input 전체를 offline 재현 가능", F_SMALL, C_NEW_B)

    # ══════════════════ 하단: Debug Panel (신규, 모니터링 레이어) ══════════════════
    dp_top, dp_h = 1750, 190
    d.rounded_rectangle((60, dp_top, 1440, dp_top + dp_h), radius=20, outline=C_NEW_B, width=3)
    for yy in range(dp_top, dp_top + dp_h, 14):
        d.line([(60, yy), (60, min(yy + 7, dp_top + dp_h))], fill=C_NEW_B, width=3)
        d.line([(1440, yy), (1440, min(yy + 7, dp_top + dp_h))], fill=C_NEW_B, width=3)
    text_center(d, 750, dp_top + 34, "Debug Panel (신규) — 전체 파이프라인을 관측하는 모니터링 레이어", font(24, bold=True), C_NEW_B)
    text_center(d, 750, dp_top + 70,
                "주행 전 사람이 눈으로 확인: fix_quality · GPS/IMU/Final heading · route bearing 차이",
                F_SMALL, C_TEXT_SUB)
    text_center(d, 750, dp_top + 98,
                "예측 궤적 · target waypoint · linear/angular · loop Hz · /control latency",
                F_SMALL, C_TEXT_SUB)

    # 모니터링 대상 박스들에서 대시보드로 내려오는 화살표는 화면이 너무 복잡해지므로
    # 대표로 몇 개만 짧게 표시 (전체 나열은 위 텍스트로 대체)
    monitor_srcs = [b_fh, b_map, b_calc]
    for src in monitor_srcs:
        sx1, sy1, sx2, sy2 = src
        bx = (sx1 + sx2) / 2
        arrow(d, (bx, sy2), (bx, dp_top), color=C_NEW_B, dashed=True, width=2)

    text_center(d, 750, dp_top + dp_h - 26,
                "→ GO 누르기 전, 이 값들이 전부 말이 되는지 사람이 먼저 확인",
                F_SMALL, C_NEW_B)

    d.text((60, dp_top + dp_h + 40),
           "즉, 이번 수정의 핵심: 주행 전에 Debug Panel로 확인 → Dry-run Gate로 실제 actuator 출력을 분리 → "
           "Replay Logger로 model input부터 control까지 tick 단위로 재현 가능하게 만듦.",
           font=F_SUB, fill=C_TEXT)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "pipeline_diagram.png"
    img.save(out_path)
    print(f"saved: {out_path}  size={img.size}")


if __name__ == "__main__":
    main()
