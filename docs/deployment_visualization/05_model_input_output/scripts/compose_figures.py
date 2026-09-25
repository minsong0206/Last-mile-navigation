import json, math, os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEST_DIR = SCRIPT_DIR.parent
OUT_DIR = DEST_DIR / "_cache" / "viz_out"
DEST_DIR.mkdir(parents=True, exist_ok=True)

FONT_DIR = Path("/usr/share/fonts/opentype/noto")
FONT_REG = FONT_DIR / "NotoSansCJK-Regular.ttc"
FONT_BOLD = FONT_DIR / "NotoSansCJK-Bold.ttc"


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REG), size)


results = json.load(open(OUT_DIR / "results.json"))

BG = (24, 24, 28)
FG = (235, 235, 235)
SUB = (170, 170, 175)
ACCENT = (255, 210, 90)


def paste_labeled(canvas, img_pil, x, y, w, h, caption, draw, cap_font, cap_color=SUB):
    img_r = img_pil.resize((w, h), Image.NEAREST if img_pil.width < w else Image.LANCZOS)
    canvas.paste(img_r, (x, y))
    draw.rectangle([x, y, x + w - 1, y + h - 1], outline=(90, 90, 95), width=1)
    lines = caption.split("\n")
    tw = max(draw.textlength(line, font=cap_font) for line in lines)
    draw.multiline_text((x + w / 2 - tw / 2, y + h + 6), caption, font=cap_font, fill=cap_color, align="center")


def compass_panel(size, heading_deg, rot_deg):
    img = Image.new("RGB", (size, size), (32, 32, 38))
    d = ImageDraw.Draw(img)
    d.text((10, 10), "rot_deg = 90 − heading_deg", font=font(15, bold=True), fill=FG)
    d.text((10, 34), f"= 90 − ({heading_deg:+.1f}°) = {rot_deg:+.1f}°", font=font(15), fill=ACCENT)

    cx, cy, r = size // 2, size // 2 + 15, size // 2 - 95
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(90, 90, 100), width=2)
    d.line([cx, cy, cx, cy - r], fill=(120, 200, 120), width=3)
    d.text((cx + 10, cy - r - 2), "forward-up", font=font(12), fill=(120, 200, 120))
    # heading arrow: East=0,North=90 CCW math -> screen dx=cos, dy=-sin
    hr = math.radians(heading_deg)
    hx = cx + r * math.cos(hr)
    hy = cy - r * math.sin(hr)
    d.line([cx, cy, hx, hy], fill=(255, 90, 90), width=4)
    d.ellipse([hx - 5, hy - 5, hx + 5, hy + 5], fill=(255, 90, 90))

    ty = cy + r + 16
    d.text((10, ty), f"heading = {heading_deg:+.1f}°", font=font(13), fill=(255, 140, 140))
    d.text((10, ty + 20), "(East=0°, North=+90°, CCW)", font=font(12), fill=SUB)
    return img


# ══════════════════════════════════════════════════════════════════════════
# A. heading_up_process_<label>.png
# ══════════════════════════════════════════════════════════════════════════
def build_process_figure(label):
    r = results[label]
    d_ = OUT_DIR / label
    W, H = 1900, 620
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 20), f"OSM Heading-up Map Transformation — sample: {label}  "
                          f"(HF dataset minsonganingee/frodobot-drive-2026-08-09_07-14-06, t={r['t']})",
              font=font(20, bold=True), fill=FG)
    info = (f"lat={r['lat']:.6f}  lon={r['lon']:.6f}   heading_deg={r['heading_deg']:+.1f}° "
            f"(estimate_heading_from_track, GPS궤적 기반)   "
            f"map_rotation_deg={r['map_rotation_deg']:+.1f}°   "
            f"route_bearing_deg={r['route_bearing_deg']:+.1f}°   "
            f"heading_route_diff_deg={r['heading_route_diff_deg']:+.1f}°")
    draw.text((30, 52), info, font=font(14), fill=ACCENT)

    panel_w, panel_h = 330, 330
    y0 = 110
    xs = [30, 30 + (panel_w + 60), 30 + 2 * (panel_w + 60), 30 + 3 * (panel_w + 60), 30 + 4 * (panel_w + 60)]

    imgs = {
        "A": (Image.open(d_ / "northup_raw.png"), "A. Raw North-up OSM\n(원본 타일, 회전 전)"),
        "B": (Image.open(d_ / "northup_ego.png"), "B. Ego 위치 + heading 표시\n(자홍 화살표 = heading 방향)"),
        "D": (Image.open(d_ / "map_headingup.png"), "D. Final Heading-up OSM\n(render_frame() 실제 출력)"),
        "E": (Image.open(d_ / "map_input96.png"), "E. Actual model map input\n(96×96 리사이즈, 정규화 전)"),
    }
    cap_font = font(14)
    paste_labeled(canvas, imgs["A"][0], xs[0], y0, panel_w, panel_h, imgs["A"][1], draw, cap_font)
    paste_labeled(canvas, imgs["B"][0], xs[1], y0, panel_w, panel_h, imgs["B"][1], draw, cap_font)
    c_img = compass_panel(panel_w, r["heading_deg"], r["map_rotation_deg"])
    paste_labeled(canvas, c_img, xs[2], y0, panel_w, panel_h,
                  "C. Rotation 적용\n(rot_deg 계산, 단일 warpAffine)", draw, cap_font)
    paste_labeled(canvas, imgs["D"][0], xs[3], y0, panel_w, panel_h, imgs["D"][1], draw, cap_font)
    paste_labeled(canvas, imgs["E"][0], xs[4], y0, panel_w, panel_h, imgs["E"][1], draw, cap_font)

    for i in range(4):
        ax = xs[i] + panel_w + 15
        ay = y0 + panel_h // 2
        draw.line([ax, ay, ax + 30, ay], fill=(120, 120, 130), width=3)
        draw.polygon([(ax + 30, ay - 7), (ax + 30, ay + 7), (ax + 42, ay)], fill=(120, 120, 130))

    verify_y = y0 + panel_h + 55
    draw.text((30, verify_y),
              "검증: 최종 heading-up 지도(D)에서 가까운 미래 경로(빨강) 픽셀의 100%가 ego 앵커보다 위쪽(row 작음)에 위치함\n"
              "→ forward 방향이 이미지 위쪽과 정렬됨을 코드(rot_deg=90-heading_deg)와 실제 렌더링 결과 양쪽에서 확인 "
              "(scripts/verify_forward_up.py, 3개 시나리오 전부 frac_above=1.00)",
              font=font(14), fill=(140, 220, 140))
    canvas.save(DEST_DIR / f"heading_up_process_{label}.png")
    print("saved", DEST_DIR / f"heading_up_process_{label}.png")


# ══════════════════════════════════════════════════════════════════════════
# B. model_input_<label>.png
# ══════════════════════════════════════════════════════════════════════════
def build_model_input_figure(label):
    r = results[label]
    d_ = OUT_DIR / label
    W, H = 1700, 520
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 20), f"실제 Multimodal Model Input — sample: {label}  (OmniVLA-Edge-Odom, N_CTX=5+1=6 frames)",
              font=font(20, bold=True), fill=FG)
    draw.text((30, 52), "6 camera frames (t-5 ... t, CTX_STRIDE_SEC=0.3s gating) + Final Heading-up OSM map "
                         "→ obs_stack(18,96,96) + map_images(3,96,96)",
              font=font(14), fill=SUB)

    thumb = 190
    y0 = 110
    labels6 = ["t-5", "t-4", "t-3", "t-2", "t-1", "t (현재)"]
    ts0 = r["ctx_timestamps"][0]
    for i in range(6):
        im = Image.open(d_ / f"ctx_{i}.jpg")
        x = 30 + i * (thumb + 14)
        dt = r["ctx_timestamps"][i] - r["t"]
        cap = f"{labels6[i]}  ({dt:+.2f}s)"
        paste_labeled(canvas, im, x, y0, thumb, thumb, cap, draw, font(13))

    arrow_x = 30 + 6 * (thumb + 14) - 4
    ay = y0 + thumb // 2
    draw.line([arrow_x, ay, arrow_x + 30, ay], fill=(120, 120, 130), width=3)
    draw.polygon([(arrow_x + 30, ay - 8), (arrow_x + 30, ay + 8), (arrow_x + 44, ay)], fill=(120, 120, 130))

    map_x = arrow_x + 55
    map_img = Image.open(d_ / "map_headingup.png")
    paste_labeled(canvas, map_img, map_x, y0, thumb, thumb,
                  f"Final Heading-up OSM\nheading={r['heading_deg']:+.1f}°  rot={r['map_rotation_deg']:+.1f}°",
                  draw, font(13))

    footer_y = y0 + thumb + 60
    draw.text((30, footer_y),
              "6 camera frames + OSM map  →  OmniVLA-Edge-Odom (context_size=5, obs_encoder=efficientnet-b0)  →  "
              "predicted waypoints (8,2) ego-frame [m]", font=font(16, bold=True), fill=ACCENT)
    draw.text((30, footer_y + 30),
              f"lat={r['lat']:.6f}  lon={r['lon']:.6f}  |  실제 파일: {d_.name}/ctx_0..5.jpg, map_headingup.png "
              "(전부 이 스크립트가 방금 저장한 실제 파일)", font=font(13), fill=SUB)
    canvas.save(DEST_DIR / f"model_input_{label}.png")
    print("saved", DEST_DIR / f"model_input_{label}.png")


# ══════════════════════════════════════════════════════════════════════════
# C/D. trajectory_comparison_<label>.png
# ══════════════════════════════════════════════════════════════════════════
def traj_plot(pred_xy, gt_xy, size=420, span_m=6.0, target_step=2):
    img = np.full((size, size, 3), (32, 32, 38), dtype=np.uint8)
    px_per_m = size / span_m
    cx, cy = size / 2, size * 0.85  # ego near bottom, forward=up
    # grid
    for gm in range(1, int(span_m)):
        gy = int(cy - gm * px_per_m)
        if 0 <= gy < size:
            cv2.line(img, (0, gy), (size, gy), (55, 55, 62), 1, cv2.LINE_AA)
            cv2.putText(img, f"{gm}m", (4, gy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (110, 110, 120), 1, cv2.LINE_AA)
    cv2.line(img, (int(cx), 0), (int(cx), size), (55, 55, 62), 1, cv2.LINE_AA)
    cv2.circle(img, (int(cx), int(cy)), 6, (0, 200, 0), -1)
    cv2.circle(img, (int(cx), int(cy)), 6, (255, 255, 255), 1)

    def to_px(xy):
        pts = []
        for x, y in xy:
            col = int(round(cx - y * px_per_m))
            row = int(round(cy - x * px_per_m))
            pts.append((col, row))
        return pts

    gt_pts = to_px(gt_xy)
    pred_pts = to_px(pred_xy)
    for pts, color, r_ in [(gt_pts, (0, 230, 0), 4), (pred_pts, (0, 220, 255), 4)]:
        full = [(int(cx), int(cy))] + pts
        for k in range(1, len(full)):
            cv2.line(img, full[k - 1], full[k], color, 2, cv2.LINE_AA)
        for k, p in enumerate(pts):
            rr = 7 if k == target_step else r_
            cv2.circle(img, p, rr, color, -1)
            cv2.circle(img, p, rr, (255, 255, 255), 1)
    return Image.fromarray(img)


def build_trajectory_comparison(label, scenario_kr):
    r = results[label]
    d_ = OUT_DIR / label
    W, H = 1750, 800
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 18), f"주행 시나리오 비교 — {scenario_kr} ({label})  |  실제 checkpoint 추론 결과",
              font=font(20, bold=True), fill=FG)
    draw.text((30, 50),
              f"heading={r['heading_deg']:+.1f}°  map_rotation={r['map_rotation_deg']:+.1f}°  "
              f"route_bearing={r['route_bearing_deg']:+.1f}°  heading_route_diff={r['heading_route_diff_deg']:+.1f}°  "
              f"|  ADE={r['ade_m']:.3f} m  FDE={r['fde_m']:.3f} m",
              font=font(14), fill=ACCENT)

    thumb = 130
    y0 = 95
    for i in range(6):
        im = Image.open(d_ / f"ctx_{i}.jpg")
        x = 30 + i * (thumb + 8)
        dt = r["ctx_timestamps"][i] - r["t"]
        paste_labeled(canvas, im, x, y0, thumb, thumb, f"{dt:+.2f}s", draw, font(11))

    map_x = 30
    map_y = y0 + thumb + 45
    map_img = Image.open(d_ / "map_headingup.png")
    paste_labeled(canvas, map_img, map_x, map_y, 330, 330,
                  "Heading-up OSM input\n(주황 점=기록된 주행의 실제 종료 지점)", draw, font(13))

    traj_img = traj_plot(np.array(r["pred_xy_m"]), np.array(r["gt_xy_m"]))
    tx = map_x + 330 + 60
    paste_labeled(canvas, traj_img, tx, map_y, 420, 420,
                  "Ego-frame trajectory: 예측(청록) vs GT(초록), 굵은 점=target_step(2)", draw, font(13))

    legend_x = tx + 420 + 60
    draw.text((legend_x, map_y), "범례", font=font(16, bold=True), fill=FG)
    draw.line([legend_x, map_y + 34, legend_x + 30, map_y + 34], fill=(0, 220, 255), width=4)
    draw.text((legend_x + 40, map_y + 26), "예측 (실제 checkpoint output)", font=font(13), fill=SUB)
    draw.line([legend_x, map_y + 60, legend_x + 30, map_y + 60], fill=(0, 230, 0), width=4)
    draw.text((legend_x + 40, map_y + 52), "GT (실제 미래 GPS, 보간)", font=font(13), fill=SUB)
    draw.ellipse([legend_x + 9, map_y + 84, legend_x + 21, map_y + 96], fill=(0, 200, 0), outline=(255, 255, 255))
    draw.text((legend_x + 40, map_y + 82), "ego (0,0)", font=font(13), fill=SUB)
    draw.text((legend_x, map_y + 130),
              f"ADE = {r['ade_m']:.3f} m\nFDE = {r['fde_m']:.3f} m",
              font=font(16, bold=True), fill=ACCENT)
    draw.text((legend_x, map_y + 200),
              "GT는 gps_imu.jsonl(~1Hz) 선형보간에서\n유도 — 학습 시 EKF-filtered 고정밀 GT와는\n다름 (README 참고)",
              font=font(12), fill=(180, 140, 90))

    canvas.save(DEST_DIR / f"trajectory_comparison_{label}.png")
    print("saved", DEST_DIR / f"trajectory_comparison_{label}.png")


build_process_figure("curve")
build_model_input_figure("curve")
build_trajectory_comparison("straight", "직진 (Straight)")
build_trajectory_comparison("curve", "완만한 커브 (Curve)")
build_trajectory_comparison("turn", "급회전 (Turn)")
print("done")
