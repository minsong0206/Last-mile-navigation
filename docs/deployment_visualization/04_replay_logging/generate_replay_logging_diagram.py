"""
generate_replay_logging_diagram.py

캡스톤 발표용 "tick_id 하나로 model input -> control까지 재현" 다이어그램.

사용한 데이터 (전부 실제 파일/로그 값, 임의 생성 없음):
  run_id = 20260925_165933 (2026-09-25 offline_smoke_test.py, dry_run=True)
  tick_id = 11
  - context_frame_paths 6개: deployment/logs/frames/20260925_165933/ctx_0000{04..09}.jpg
    (2026-09-18 실배포 스크린캐스트에서 추출한 실제 카메라 프레임)
  - map_replay_path: .../map_000011.png (실제 저장된 model-input map,
    예측 오버레이는 실제 pred_xy_m으로 이 스크립트가 다시 그림 — 저장 당시와
    동일한 LiveMapBuilder.draw_predicted_trajectory() 사용)
  - pred_xy_m / target_waypoint_xy / (계산된) linear,angular: 전부 해당 tick의
    JSONL step/control_sent 레코드 값 그대로

실행: (frodobot conda env, 외부 연결 불필요)
    python3 docs/deployment_visualization/04_replay_logging/generate_replay_logging_diagram.py
출력: docs/deployment_visualization/04_replay_logging/replay_logging_diagram.png
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "deployment"))
sys.path.insert(0, str(REPO_ROOT / "osm_pipeline" / "py"))
from build_live_map import LiveMapBuilder

OUT_DIR = Path(__file__).resolve().parent
LOG_DIR = REPO_ROOT / "deployment" / "logs"
RUN_ID = "20260925_165933"
TICK_ID = 11

FONT_DIR = Path("/usr/share/fonts/opentype/noto")
FONT_REG = FONT_DIR / "NotoSansCJK-Regular.ttc"
FONT_BOLD = FONT_DIR / "NotoSansCJK-Bold.ttc"


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REG), size)


C_TEXT = (17, 24, 39)
C_SUB = (75, 85, 99)
C_NEW_B = (194, 65, 12)
C_NEW = (255, 237, 213)
C_MODEL = (237, 233, 254)
C_MODEL_B = (109, 40, 217)
C_OLD_B = (156, 163, 175)


def load_tick():
    lines = [json.loads(l) for l in open(LOG_DIR / f"deploy_{RUN_ID}.jsonl")]
    step = next(l for l in lines if l.get("type") == "step" and l.get("tick_id") == TICK_ID)
    ctrl = next(l for l in lines if l.get("type") == "control_sent" and abs(l["ts"] - step["ts"]) < 1.0)
    return step, ctrl


def rrect(d, xy, fill=None, outline=None, width=2, radius=12):
    d.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def arrow(d, p0, p1, color=(55, 65, 81), width=3):
    import math
    d.line([p0, p1], fill=color, width=width)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    for da in (0.5, -0.5):
        a2 = ang + math.pi - da
        d.line([p1, (p1[0] + 11 * math.cos(a2), p1[1] + 11 * math.sin(a2))], fill=color, width=width)


def main():
    step, ctrl = load_tick()
    F_TITLE = font(38, bold=True)
    F_SUB = font(20)
    F_LABEL = font(22, bold=True)
    F_SMALL = font(17)
    F_MONO = font(19)

    W, H = 2100, 1500
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)

    d.text((60, 30), "Exact Model-input Replay — tick_id 하나로 그 순간 입력 전체를 재현",
           font=F_TITLE, fill=C_TEXT)
    d.text((60, 82), f"run_id={RUN_ID}   tick_id={TICK_ID}   (deployment/logs/frames/{RUN_ID}/ 실제 파일 사용, 가상 데이터 없음)",
           font=F_SUB, fill=C_SUB)

    # ── tick_id 배지 ──
    rrect(d, (60, 150, 260, 220), fill=C_NEW, outline=C_NEW_B, width=3)
    d.text((90, 168), f"tick_id = {TICK_ID}", font=F_LABEL, fill=C_NEW_B)

    # ── 6 카메라 프레임 (context_frame_paths) ──
    thumb_w, thumb_h = 190, 140
    x0 = 60
    y0 = 280
    d.text((x0, y0 - 34), "context_frame_paths (6장, 새 프레임일 때만 저장·dedup)", font=F_LABEL, fill=C_TEXT)
    frame_paths = step["context_frame_paths"]
    thumbs_x = []
    for i, rel in enumerate(frame_paths):
        im = Image.open(LOG_DIR / rel).convert("RGB")
        im.thumbnail((thumb_w, thumb_h))
        px = x0 + i * (thumb_w + 14)
        img.paste(im, (px, y0))
        d.rectangle((px, y0, px + im.width, y0 + im.height), outline=C_OLD_B, width=2)
        d.text((px, y0 + thumb_h + 4), Path(rel).name, font=F_SMALL, fill=C_SUB)
        thumbs_x.append(px + im.width // 2)

    # ── model-input map (예측 오버레이 다시 그림, 저장 당시와 동일 함수) ──
    map_x = x0 + 6 * (thumb_w + 14) + 30
    map_np = np.array(Image.open(LOG_DIR / step["map_replay_path"]).convert("RGB"))
    builder = LiveMapBuilder(map_range_m=20.0)
    pred_xy_m = np.array(step["pred_xy_m"])
    overlay = builder.draw_predicted_trajectory(map_np, pred_xy_m)
    map_img = Image.fromarray(overlay)
    map_img.thumbnail((220, 220))
    d.text((map_x, y0 - 34), "model-input map (+예측 궤적)", font=F_LABEL, fill=C_TEXT)
    img.paste(map_img, (map_x, y0))
    d.rectangle((map_x, y0, map_x + map_img.width, y0 + map_img.height), outline=C_NEW_B, width=3)
    d.text((map_x, y0 + map_img.height + 4), Path(step["map_replay_path"]).name, font=F_SMALL, fill=C_SUB)

    # ── 화살표 -> OmniVLA ──
    model_cx, model_cy = 1000, 720
    for tx in thumbs_x + [map_x + map_img.width // 2]:
        arrow(d, (tx, y0 + thumb_h + 30), (model_cx - 140, model_cy - 40), color=(180, 180, 190), width=1)
    rrect(d, (model_cx - 160, model_cy - 60, model_cx + 160, model_cy + 60), fill=C_MODEL, outline=C_MODEL_B, width=3)
    d.text((model_cx - 100, model_cy - 20), "OmniVLA-Edge", font=F_LABEL, fill=C_TEXT)

    # ── Predicted waypoints (간단한 궤적 플롯) ──
    plot_x, plot_y, plot_w, plot_h = 1260, 620, 300, 220
    rrect(d, (plot_x, plot_y, plot_x + plot_w, plot_y + plot_h), outline=C_OLD_B, width=2)
    d.text((plot_x, plot_y - 30), "Predicted Waypoints (pred_xy_m)", font=F_LABEL, fill=C_TEXT)
    xs = pred_xy_m[:, 0]
    ys = pred_xy_m[:, 1]
    max_x = max(xs.max(), 1e-3)
    max_y = max(abs(ys).max(), 1e-3)
    cx0 = plot_x + 30
    cy0 = plot_y + plot_h / 2
    scale_x = (plot_w - 60) / max_x
    scale_y = (plot_h / 2 - 20) / max_y
    pts = [(cx0 + x * scale_x, cy0 - y * scale_y) for x, y in zip(xs, ys)]
    d.line([(cx0, plot_y + 10), (cx0, plot_y + plot_h - 10)], fill=(230, 230, 230), width=1)
    d.line([(plot_x + 10, cy0), (plot_x + plot_w - 10, cy0)], fill=(230, 230, 230), width=1)
    d.line(pts, fill=(6, 182, 212), width=3)
    for k, p in enumerate(pts):
        r = 6 if k == 2 else 3
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=(6, 182, 212), outline=(255, 255, 255))
    d.ellipse([cx0 - 5, cy0 - 5, cx0 + 5, cy0 + 5], fill=(30, 150, 60))
    arrow(d, (model_cx + 160, model_cy), (plot_x - 10, plot_y + plot_h / 2))

    # ── target waypoint / controller / calculated control ──
    info_x, info_y = 1630, 620
    rrect(d, (info_x, info_y, info_x + 400, info_y + 260), fill=(255, 255, 255), outline=C_OLD_B, width=2)
    d.text((info_x + 20, info_y + 16), "target_waypoint_xy", font=F_LABEL, fill=C_TEXT)
    tw = step["target_waypoint_xy"]
    d.text((info_x + 20, info_y + 50), f"({tw[0]:+.3f}, {tw[1]:+.3f}) m", font=F_MONO, fill=C_SUB)
    d.text((info_x + 20, info_y + 100), "Controller 계산 결과", font=F_LABEL, fill=C_TEXT)
    d.text((info_x + 20, info_y + 134), f"linear  = {ctrl['computed_linear']:+.3f} m/s", font=F_MONO, fill=C_SUB)
    d.text((info_x + 20, info_y + 164), f"angular = {ctrl['computed_angular']:+.3f} rad/s", font=F_MONO, fill=C_SUB)
    d.text((info_x + 20, info_y + 204), f"http_status={ctrl['http_status']}  dry_run={ctrl['dry_run']}", font=F_SMALL, fill=C_SUB)
    arrow(d, (plot_x + plot_w + 10, plot_y + plot_h / 2), (info_x - 10, info_y + 60))

    # ── 하단: 0918 vs 지금 비교 콜아웃 ──
    y = 1000
    rrect(d, (60, y, 2040, y + 420), outline=C_NEW_B, width=3)
    d.text((90, y + 20), "0918 실배포 당시 vs 지금 — 무엇이 달라졌는가", font=font(28, bold=True), fill=C_NEW_B)

    col_w = (2040 - 60 - 60) / 2
    lx = 90
    d.text((lx, y + 80), "2026-09-18 (당시)", font=F_LABEL, fill=(107, 114, 128))
    old_lines = [
        "대시보드 카메라(/frame.jpg)는 배포 루프의 frame_buffer와",
        "독립적인 주기로 갱신됨 — \"화면에 보이던 것\" ≠ \"그 순간",
        "모델에 실제로 들어간 6프레임\"",
        "",
        "그래서 나중에 angular=+0.3 포화 순간을 대시보드 화면으로",
        "재구성해봤지만, 그날 실제 기록된 pred_xy_m과 전혀 다른",
        "예측이 나옴 → 원인(heading/geometry/카메라) 분리 불가",
        "→ \"재현 불가\"로 결론(2026-09-25 세션에서 직접 검증)",
    ]
    yy = y + 120
    for ln in old_lines:
        d.text((lx, yy), ln, font=F_SMALL, fill=(107, 114, 128))
        yy += 30

    rx = 90 + col_w + 60
    d.text((rx, y + 80), "지금 (Replay Logger 도입 후)", font=F_LABEL, fill=C_NEW_B)
    new_lines = [
        "predict_waypoints()가 실제로 obs_stack/map_tensor를 만드는 데",
        "쓴 원본 이미지를 tick_id로 그대로 저장(context_frame_paths,",
        "map_replay_path) — 대시보드와 무관한 별도 기록",
        "",
        "이 그림 자체가 증거: tick_id=11 하나로 그 순간의 6프레임 +",
        "지도 + 예측 + target waypoint + 계산된 control까지 전부",
        "다시 열어서 확인 가능 (offline_smoke_test.py로 매 항목 검증됨)",
    ]
    yy = y + 120
    for ln in new_lines:
        d.text((rx, yy), ln, font=F_SMALL, fill=C_NEW_B)
        yy += 30

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "replay_logging_diagram.png"
    img.save(out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
