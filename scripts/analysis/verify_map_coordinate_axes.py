"""
verify_map_coordinate_axes.py

목적: osm_map_generator.py의 render_frame()이 실제로 "위=전진 방향, 오른쪽=로봇 우측"
     이 맞는지 합성(synthetic) 입력으로 검증한다. 실제 OSM 타일 서버(네트워크) 불필요 —
     캔버스를 직접 나침반 라벨(N/S/E/W)로 채워서 render_frame()에 그대로 통과시킨다.

Test 1 (compass): 여러 heading(East/North/West/South)에서 캔버스에 그린 N/S/E/W 텍스트가
     회전 후 어느 위치에 오는지 확인. heading=East(0rad)로 로봇이 동쪽을 보고 있다면,
     "내가 지금 보고 있는 방향(E)"은 이미지 위쪽에, 왼쪽으로 90도 튼 방향(N)은 이미지
     왼쪽에, 오른쪽으로 90도 튼 방향(S)은 이미지 오른쪽에 와야 정상이다.

Test 2 (route curve): heading=East에서 실제 훈련 데이터처럼 "미래 경로(빨간선)"가
     남쪽(S)으로 휘어지는 경로(= 로봇 기준 우회전)와 북쪽(N)으로 휘어지는 경로
     (= 로봇 기준 좌회전)를 render_frame()에 통과시켜, 우회전 경로가 실제로 이미지
     오른쪽으로 그려지는지 확인.

실행 (host, 네트워크 불필요):
  python3 scripts/analysis/verify_map_coordinate_axes.py
출력: attention_analysis/coord_axis_test_*.png
"""

import sys
import math
from pathlib import Path

import numpy as np
import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "osm_pipeline" / "py"))
from osm_map_generator import render_frame, latlon_to_pixel_global, meters_per_pixel, ZOOM

OUT_DIR = REPO_ROOT / "attention_analysis"
OUT_DIR.mkdir(exist_ok=True)

# 임의의 기준 위경도 (실제 위치는 중요하지 않음 — 순수 좌표축 검증용)
LAT0, LON0 = 37.5, 127.0
ZOOM_T = ZOOM  # 18
MAP_RANGE_M = 40.0  # 라벨을 넉넉히 배치하기 위해 기본값(25m)보다 살짝 키움


def make_compass_canvas(lat_curr, lon_curr, zoom, label_radius_m=25.0, canvas_half_px=400):
    """ego 위치를 중심으로 흰 캔버스를 만들고, 실제 지리적 N/S/E/W 방향에
    텍스트 라벨을 찍는다. 캔버스는 표준 슬리피맵 타일과 동일하게
    North-up(위=북, 오른쪽=동) 픽셀 좌표계를 따른다."""
    ego_gx, ego_gy = latlon_to_pixel_global(lat_curr, lon_curr, zoom)
    gx0 = ego_gx - canvas_half_px
    gy0 = ego_gy - canvas_half_px
    canvas = np.ones((canvas_half_px * 2, canvas_half_px * 2, 3), dtype=np.uint8) * 255

    mpp = meters_per_pixel(lat_curr, zoom)
    r_px = int(label_radius_m / mpp)
    cx, cy = canvas_half_px, canvas_half_px

    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = {
        "N": (cx, cy - r_px),   # 북 = 캔버스 위쪽 (y 감소)
        "S": (cx, cy + r_px),   # 남 = 캔버스 아래쪽 (y 증가)
        "E": (cx + r_px, cy),   # 동 = 캔버스 오른쪽 (x 증가)
        "W": (cx - r_px, cy),   # 서 = 캔버스 왼쪽 (x 감소)
    }
    colors = {"N": (0, 0, 200), "S": (200, 0, 0), "E": (0, 150, 0), "W": (150, 0, 150)}
    for label, (px, py) in labels.items():
        cv2.circle(canvas, (px, py), 6, colors[label], -1)
        cv2.putText(canvas, label, (px - 12, py - 14), font, 1.1, colors[label], 3, cv2.LINE_AA)
    # 격자선 (참고용)
    cv2.line(canvas, (cx, 0), (cx, canvas_half_px * 2), (220, 220, 220), 1)
    cv2.line(canvas, (0, cy), (canvas_half_px * 2, cy), (220, 220, 220), 1)

    return canvas, gx0, gy0


def run_compass_test():
    print("=" * 70)
    print("TEST 1: Compass label rotation test (heading별 N/S/E/W 위치 확인)")
    print("=" * 70)

    # heading_rad: East=0, North=+90deg, West=180deg, South=-90deg (코드 주석 기준)
    headings = {
        "heading=East(facing east, 0deg)": 0.0,
        "heading=North(facing north, 90deg)": math.radians(90),
        "heading=West(facing west, 180deg)": math.radians(180),
        "heading=South(facing south, -90deg)": math.radians(-90),
    }

    canvas, gx0, gy0 = make_compass_canvas(LAT0, LON0, ZOOM_T, label_radius_m=25.0)

    results = {}
    for name, heading_rad in headings.items():
        img = render_frame(
            canvas, gx0, gy0, ZOOM_T,
            LAT0, LON0, heading_rad,
            future_lats=None, future_lons=None,
            past_lats=None, past_lons=None,
            out_size=480, map_range_m=MAP_RANGE_M,
        )
        results[name] = img
        out_path = OUT_DIR / f"coord_axis_test_compass_{name.split('(')[0].split('=')[1]}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"  saved: {out_path}")

    # 2x2 합성 이미지
    keys = list(headings.keys())
    top = np.hstack([results[keys[0]], results[keys[1]]])
    bot = np.hstack([results[keys[2]], results[keys[3]]])
    combined = np.vstack([top, bot])
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    # 라벨 추가
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = results[keys[0]].shape[:2]
    for i, k in enumerate(keys):
        x = (i % 2) * w + 10
        y = (i // 2) * h + 30
        cv2.putText(combined_bgr, k, (x, y), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    out_path = OUT_DIR / "coord_axis_test_compass_combined.png"
    cv2.imwrite(str(out_path), combined_bgr)
    print(f"\n  → 합성 이미지: {out_path}")
    print("""
  기대 결과 (rot_deg = 90 - heading_deg 공식이 맞다면):
    - heading=East(0deg) 로봇이 동쪽을 보고 있음 → 이미지 위쪽=E, 왼쪽=N, 오른쪽=S, 아래쪽=W
      (동쪽을 보고 있을 때 왼쪽으로 90도 틀면 북쪽 → N이 왼쪽에 와야 정상)
    - heading=North(90deg) → 위쪽=N, 왼쪽=W, 오른쪽=E, 아래쪽=S
    - heading=West(180deg) → 위쪽=W, 왼쪽=S, 오른쪽=N, 아래쪽=E
    - heading=South(-90deg) → 위쪽=S, 왼쪽=E, 오른쪽=W, 아래쪽=N
  즉 항상: 위=현재 heading 방향, 왼쪽=heading에서 반시계 90도(로봇의 좌측),
           오른쪽=heading에서 시계 90도(로봇의 우측) 가 되어야 정상입니다.
    """)


def _meters_to_latlon_offset(lat0, lon0, dx_east_m, dy_north_m):
    LAT_M = 111320.0
    dlat = dy_north_m / LAT_M
    dlon = dx_east_m / (LAT_M * math.cos(math.radians(lat0)))
    return lat0 + dlat, lon0 + dlon


def _make_curving_path(lat0, lon0, start_heading_deg, turn_rate_deg_per_step, n_steps=10, step_m=3.0):
    """heading에서 시작해 매 스텝 turn_rate_deg_per_step만큼 꺾이는 경로를 생성.
    turn_rate < 0 → 시계방향(오른쪽으로 회전), turn_rate > 0 → 반시계방향(왼쪽으로 회전)."""
    x, y = 0.0, 0.0
    heading = start_heading_deg
    lats, lons = [], []
    for _ in range(n_steps):
        heading += turn_rate_deg_per_step
        rad = math.radians(heading)
        x += step_m * math.cos(rad)  # East component
        y += step_m * math.sin(rad)  # North component
        lat, lon = _meters_to_latlon_offset(lat0, lon0, x, y)
        lats.append(lat)
        lons.append(lon)
    return np.array(lats), np.array(lons)


def run_route_curve_test():
    print("=" * 70)
    print("TEST 2: Future route curve test (heading=East, 우회전 vs 좌회전 경로)")
    print("=" * 70)

    canvas, gx0, gy0 = make_compass_canvas(LAT0, LON0, ZOOM_T, label_radius_m=25.0)
    heading_rad = 0.0  # East

    # 오른쪽으로 도는 경로 (heading 감소 = 시계방향 = 로봇 기준 우회전)
    right_lats, right_lons = _make_curving_path(LAT0, LON0, start_heading_deg=0, turn_rate_deg_per_step=-8)
    # 왼쪽으로 도는 경로 (heading 증가 = 반시계방향 = 로봇 기준 좌회전)
    left_lats, left_lons = _make_curving_path(LAT0, LON0, start_heading_deg=0, turn_rate_deg_per_step=+8)

    img_right = render_frame(
        canvas, gx0, gy0, ZOOM_T, LAT0, LON0, heading_rad,
        future_lats=right_lats, future_lons=right_lons,
        past_lats=None, past_lons=None,
        out_size=480, map_range_m=MAP_RANGE_M,
    )
    img_left = render_frame(
        canvas, gx0, gy0, ZOOM_T, LAT0, LON0, heading_rad,
        future_lats=left_lats, future_lons=left_lons,
        past_lats=None, past_lons=None,
        out_size=480, map_range_m=MAP_RANGE_M,
    )
    img_straight = render_frame(
        canvas, gx0, gy0, ZOOM_T, LAT0, LON0, heading_rad,
        future_lats=right_lats[:1], future_lons=right_lons[:1],
        past_lats=None, past_lons=None,
        out_size=480, map_range_m=MAP_RANGE_M,
    )

    cv2.imwrite(str(OUT_DIR / "coord_axis_test_route_right.png"), cv2.cvtColor(img_right, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT_DIR / "coord_axis_test_route_left.png"), cv2.cvtColor(img_left, cv2.COLOR_RGB2BGR))

    combined = np.hstack([img_left, img_straight, img_right])
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    w = img_left.shape[1]
    cv2.putText(combined_bgr, "LEFT turn (should curve LEFT in image)", (10, 30), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(combined_bgr, "straight (ego only)", (w + 10, 30), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(combined_bgr, "RIGHT turn (should curve RIGHT in image)", (2 * w + 10, 30), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    out_path = OUT_DIR / "coord_axis_test_route_combined.png"
    cv2.imwrite(str(out_path), combined_bgr)
    print(f"  → 합성 이미지: {out_path}")
    print("""
  기대 결과: 빨간 미래 경로선이
    - RIGHT turn 케이스 → 이미지 오른쪽으로 휘어야 정상 (학습 데이터의 우회전 케이스와 동일 규칙)
    - LEFT  turn 케이스 → 이미지 왼쪽으로 휘어야 정상
  만약 반대로 나온다면 render_frame()의 회전 부호가 뒤집혀 있다는 뜻이고,
  이게 사실이라면 모델이 학습 내내 "실제로는 왼쪽인데 오른쪽이라고 라벨링된" 맵을
  봐온 셈이 되어, map-zero(직진)에서 우회전을 예측하는 버그를 정확히 설명할 수 있습니다.
    """)


if __name__ == "__main__":
    run_compass_test()
    run_route_curve_test()
    print(f"\n모든 결과: {OUT_DIR}")
