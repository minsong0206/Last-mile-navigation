"""
build_live_map.py

배포(FrodoBot Mini 실로봇)용 실시간 odom map 생성기.

배경: 학습 데이터(osm_map_generator.py::process_episode)는 이미 다 끝난 라이드의
      "미래 GPS 기록"을 그대로 빨간 선(future route)으로 그렸다. 실배포도 출발/도착
      GPS 좌표를 미리 알고 "길찾기 모드"로 도는 구조이므로, 학습 때와 똑같은 패턴을
      그대로 재사용한다: 경로(route)는 미리 한 번만 계산해서 고정해두고, 매 프레임은
      "현재 위치가 그 고정된 경로의 어디쯤인지" 찾아서 과거/미래로 나눠 그리기만 한다.

      ❌ 예전 버전의 문제: get_map_image()를 호출할 때마다 매번 OSRM에 새로 쿼리를
         던졌음 (제어 루프 3Hz 기준 초당 3번) — osmnav(github.com/hmmdyn/osmnav)
         구조를 참고해서 확인해보니, 그쪽은 목적지 설정 시 딱 1번만 라우팅하고
         지도 렌더링 노드는 그 결과를 구독만 함. 우리도 이 패턴으로 수정함.
      ✅ 지금 버전: set_goal()을 배포 시작 시 1번만 호출해서 경로를 계산+캐싱하고,
         get_map_image()는 캐싱된 경로에서 현재 위치와 가장 가까운 지점을 찾아
         (process_episode()의 map_frames_to_route()와 동일한 발상) 그 지점 기준으로
         과거/미래를 나눠 그리기만 한다. 매 프레임 네트워크 요청 없음.

학습 때와 동일하게 유지해야 하는 것 (osm_map_generator.py 기준):
  - MAP_RANGE_M(전방 reach)/REAR_RATIO/앵커, 회전 공식(rot_deg = 90 - heading_deg),
    heading-up 정렬, 타일 제공자(cartocdn voyager_nolabels), ZOOM=19 — 전부
    osm_map_generator.py에서 import해서 쓰므로 이 파일에 값을 복제하지 말 것
  - ROUTE_COLOR(빨강)=미래 경로, PAST_COLOR(회색)=과거 경로, ROUTE_WIDTH
  - 최종 리사이즈 96x96 + IMG_MEAN/IMG_STD 정규화 (모델 입력 직전 rides11_dataset과 동일)
  - heading_rad의 "소스"도 학습과 같아야 함 — 학습 heading은 GPS 궤적 기반(atan2)이므로
    이 함수에 IMU 컴퍼스를 그대로 넣지 말 것 (2026-08-25 실배포에서 IMU-vs-GPS 궤적
    heading이 평균 +97° 어긋남을 확인 — omnivla_edge_deploy.py는 이제
    estimate_heading_from_track()로 만든 GPS궤적 기반 heading을 넘김, IMU는 폴백 전용)

실로봇에서 학습 때와 다를 수밖에 없는 것:
  - future route가 GT 기록이 아니라 출발 전 미리 계산해둔 OSRM 경로
  - past route는 로봇이 실제로 지나온 odometry 누적 기록 (학습 때는 GT GPS, 여기선
    로봇 자체 odometry/GPS 센서 값 — 논리적으로는 동일한 종류의 데이터)

사용 예:
  builder = LiveMapBuilder(map_range_m=20.0)
  builder.set_goal(start_lat=37.5, start_lon=127.0, goal_lat=37.501, goal_lon=127.001)  # 1회만

  # 이후 매 제어 루프(3Hz)마다:
  map_tensor = builder.get_map_tensor(
      lat=37.5001, lon=127.0002, heading_rad=0.3,
      past_track=[(37.4998, 126.9995), (37.4999, 126.9998), ...],
  )
  # map_tensor: (1, 3, 96, 96) torch.Tensor, IMG_MEAN/IMG_STD 정규화 완료 — 모델에 바로 입력 가능

  # 경로 이탈이 심하면(예: 3m 이상) 재계산:
  if builder.is_off_route(lat, lon, threshold_m=3.0):
      builder.set_goal(lat, lon, goal_lat, goal_lon)
"""

import sys
import math
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from torchvision import transforms
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "osm_pipeline" / "py"))
from osm_map_generator import (
    osrm_port, build_canvas, render_frame, _densify_route,
    latlon_to_pixel_global, global_to_canvas, meters_per_pixel,
    MAP_RANGE_M as TRAIN_MAP_RANGE_M, ZOOM as TRAIN_ZOOM, MAP_SIZE_PX, REAR_RATIO,
)

# rides11_dataset.py와 동일한 정규화 (모델이 학습 때 본 것과 동일한 분포로 맞추기 위함)
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

LAT_M = 111320.0  # 위도 1도당 미터 (haversine 근사 대신 간단한 등적원통 근사, 근거리 오차 무시 가능)


def _pick_scale_bar_m(map_range_m):
    """스케일바용 '보기 좋은' 미터 값 선택 (지도 반경의 대략 절반 이하 중 가장 큰 값)."""
    candidates = [1, 2, 5, 10, 20, 50]
    fit = [c for c in candidates if c <= map_range_m * 0.6]
    return fit[-1] if fit else candidates[0]


def snap_to_nearest_road(lat, lon, port, timeout=3.0, max_snap_m=30.0):
    """OSRM /nearest로 (lat,lon)을 가장 가까운 매핑된 보행로 위 지점으로 스냅.

    2026-09-18: 로봇이 실제로 보도 위에 서 있어도, GPS 수신 오차(특히 건물 근처
    멀티패스)로 원시 좌표가 매핑된 길에서 수 m 떨어져 찍히는 경우가 실측됨
    (같은 자리에서 3.13m 오차). set_goal()이 이 원시 좌표를 그대로 경로 시작점으로
    쓰면, 경로가 실제 로봇 위치와 계속 어긋난 채로 시작돼 미래경로선에 인위적인
    꺾임이 생김(is_off_route 재라우팅 쿨다운으로도 이 어긋남 자체는 못 없앰).
    그래서 경로 계산 직전에 시작/목표 좌표를 매핑된 길 위로 스냅해서, 경로가
    로봇의 "의도된"(보도 위) 위치와 맞게 시작하도록 한다.

    max_snap_m: 스냅 거리가 이보다 크면(=근처에 매핑된 길이 아예 없음) 원본 좌표를
    그대로 반환 — 엉뚱하게 먼 지점으로 스냅하는 것을 방지.

    반환: (lat, lon, snap_distance_m) — 스냅 실패/너무 멀면 snap_distance_m=None
    (2026-09-25 추가: pre-drive 대시보드에서 스냅 거리를 직접 보여주기 위해 원본
    2-tuple 반환에서 확장함 — 값 자체나 스냅 로직은 안 바꿨음)."""
    url = f"http://localhost:{port}/nearest/v1/foot/{lon},{lat}"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        wp = r.json()["waypoints"][0]
        if wp["distance"] <= max_snap_m:
            snapped_lon, snapped_lat = wp["location"]
            return snapped_lat, snapped_lon, float(wp["distance"])
    except Exception as e:
        print(f"[build_live_map] snap_to_nearest_road 실패({e}), 원본 좌표 사용")
    return lat, lon, None


def query_osrm_route(start_lat, start_lon, goal_lat, goal_lon, port,
                      interp_step_m=1.0, timeout=3.0, max_retries=3, retry_delay=1.0):
    """출발→도착 OSRM foot 경로 쿼리. 배포 시작 시 1번만 호출됨 (osmnav의
    navigation_node.py 패턴 참고 — retry 포함). 실패 시 직선 경로로 폴백.
    반환: (route(M,2) [[lat,lon], ...] interp_step_m 간격 densify됨, used_fallback: bool)
    (2026-09-25: used_fallback 추가 — 이전엔 콘솔 print만 있고 호출 측에서 폴백
    여부를 알 방법이 없었음. pre-drive 대시보드에서 "OSRM 정상/fallback"을
    보여주려면 필요함)."""
    import time
    url = (f"http://localhost:{port}/route/v1/foot/"
           f"{start_lon},{start_lat};{goal_lon},{goal_lat}?overview=full&geometries=geojson")
    last_exc = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            d = r.json()
            raw = d["routes"][0]["geometry"]["coordinates"]  # [[lon,lat], ...]
            route = np.array([[pt[1], pt[0]] for pt in raw])
            return _densify_route(route, interp_step_m), False
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
    print(f"[build_live_map] OSRM 쿼리 {max_retries}회 실패({last_exc}), 직선 경로로 폴백")
    route = np.array([[start_lat, start_lon], [goal_lat, goal_lon]])
    return _densify_route(route, interp_step_m), True


class LiveMapBuilder:
    def __init__(self, map_range_m: float = TRAIN_MAP_RANGE_M, zoom: int = TRAIN_ZOOM,
                 out_size: int = MAP_SIZE_PX, model_input_size: int = 96):
        """
        map_range_m: 학습 때 쓴 것과 반드시 동일해야 함 (예: 20m 재학습 체크포인트를
                     쓴다면 여기도 20.0으로 맞출 것 — 다르면 학습-추론 분포 불일치).
                     2026-09 기준 이 값은 "전방 reach(m)"이지 half-width가 아님
                     (osm_map_generator.py::render_frame, REAR_RATIO 참고) — ego는
                     이미지 정중앙이 아니라 앵커(전방:후방=1:REAR_RATIO)에 위치함.
        """
        self.map_range_m = map_range_m
        self.zoom = zoom
        self.out_size = out_size
        self.session = requests.Session()
        self.transform = transforms.Compose([
            transforms.Resize((model_input_size, model_input_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ])
        self._route_latlon = None      # (M,2) [[lat,lon],...] — set_goal()에서 1회 계산
        self._canvas = None            # (canvas_bgr, gx0, gy0) — route 전체를 커버하는 캔버스
        # 2026-09-25 추가: pre-drive 대시보드에서 "route가 정상/fallback인지",
        # "스냅이 얼마나 멀리 일어났는지"를 보여주기 위한 최근 set_goal() 결과 캐시.
        self.last_osrm_fallback = None
        self.last_start_snap_m = None
        self.last_goal_snap_m = None

    def set_goal(self, start_lat, start_lon, goal_lat, goal_lon):
        """경로를 1번 계산해서 캐싱. 배포 시작 시(또는 경로 이탈 재계산 시)만 호출.
        get_map_image()는 이후 이 캐싱된 경로만 사용하고 네트워크 요청을 하지 않음."""
        port = osrm_port(start_lat, start_lon)
        snapped_start_lat, snapped_start_lon, start_snap_m = snap_to_nearest_road(start_lat, start_lon, port)
        snapped_goal_lat, snapped_goal_lon, goal_snap_m = snap_to_nearest_road(goal_lat, goal_lon, port)
        self._route_latlon, used_fallback = query_osrm_route(
            snapped_start_lat, snapped_start_lon, snapped_goal_lat, snapped_goal_lon, port)
        self._canvas = build_canvas(self._route_latlon[:, 0], self._route_latlon[:, 1],
                                     self.zoom, self.session)
        self.last_osrm_fallback = used_fallback
        self.last_start_snap_m = start_snap_m
        self.last_goal_snap_m = goal_snap_m
        print(f"[LiveMapBuilder] 경로 캐싱 완료: {len(self._route_latlon)}개 포인트 "
              f"(원시 출발=({start_lat:.5f},{start_lon:.5f}) → 스냅됨=({snapped_start_lat:.5f},{snapped_start_lon:.5f}), "
              f"목표=({snapped_goal_lat:.5f},{snapped_goal_lon:.5f}), "
              f"snap거리(시작/목표)=({start_snap_m},{goal_snap_m})m, osrm_fallback={used_fallback})")

    def get_route_latlon(self):
        """캐싱된 경로 전체를 (M,2) [[lat,lon],...] 리스트로 반환 (로그/분석용)."""
        if self._route_latlon is None:
            return None
        return self._route_latlon.tolist()

    def _closest_route_idx(self, lat, lon):
        """process_episode()의 map_frames_to_route()와 같은 발상: 현재 위치가
        캐싱된 경로의 어느 지점에 가장 가까운지 찾는다 (매 프레임 O(M), M은
        보통 수백 개 이내라 실시간 루프에 문제없음)."""
        d = np.hypot(self._route_latlon[:, 0] - lat, self._route_latlon[:, 1] - lon)
        return int(np.argmin(d)), float(d.min())

    def is_off_route(self, lat, lon, threshold_m=3.0):
        """경로에서 threshold_m 이상 벗어났는지 확인 (osmnav의 재라우팅 트리거와 동일 개념).
        True면 호출 측(배포 루프)에서 set_goal()을 다시 불러 재계산해야 함.

        2026-09-18: 기본값 15m→3m로 낮춤 — 로봇이 경로에서 몇 m만 벗어나도
        get_map_image()가 "현재(벗어난) 위치→경로 위 가장 가까운 점"을 잇느라 미래
        경로선에 인위적인 꺾임이 생기고, 그게 학습 데이터엔 없던 형태라 모델이 실제
        회전 상황으로 오인하는 문제를 실측함(deploy_20260918_183939.jsonl — 로봇이
        경로에서 옆으로 ~4m 벗어나 있을 때 지도에 꺾임이 그려지고 모델이 좌회전 예측).
        15m는 "완전히 딴 길로 샜을 때"엔 맞는 값이지만 이 정도 작은 이탈까진 못 잡음."""
        if self._route_latlon is None:
            return False
        _, dist_deg = self._closest_route_idx(lat, lon)
        dist_m = dist_deg * LAT_M  # 위경도 차 → 대략적인 미터 환산 (근거리 근사)
        return dist_m > threshold_m

    def route_bearing_rad(self, lat, lon, lookahead_m=5.0):
        """2026-09-25: 디버그 대시보드/로깅용 — 현재 위치에서 캐싱된 경로를 따라
        lookahead_m만큼 앞선 지점까지의 방향(진행해야 할 방향)을
        estimate_heading_from_track()과 동일한 공식(atan2(북쪽성분, 동쪽성분))으로
        계산한다. "heading(로봇이 향한 방향) - route_bearing(가야 할 방향)" 차이를
        보면 heading-up 지도 회전이 실제로 경로와 정렬돼있는지 수치로 확인 가능
        (0918 replay에서 이 차이가 ~41°였던 것과 비교하는 용도).
        경로가 없거나 현재 위치가 경로 끝 근처라 lookahead_m를 못 채우면 None."""
        if self._route_latlon is None:
            return None
        idx, _ = self._closest_route_idx(lat, lon)
        route = self._route_latlon
        if idx >= len(route) - 1:
            return None
        lat0, lon0 = route[idx]
        cum_m = 0.0
        j = idx
        while j < len(route) - 1 and cum_m < lookahead_m:
            j += 1
            dlat = (route[j][0] - route[j - 1][0]) * LAT_M
            dlon = (route[j][1] - route[j - 1][1]) * LAT_M * math.cos(math.radians(route[j - 1][0]))
            cum_m += math.hypot(dlat, dlon)
        dlat = (route[j][0] - lat0) * LAT_M
        dlon = (route[j][1] - lon0) * LAT_M * math.cos(math.radians(lat0))
        if math.hypot(dlat, dlon) < 1e-6:
            return None
        return math.atan2(dlat, dlon)

    def get_northup_preview_image(self, lat, lon, heading_rad=None, out_size=None):
        """2026-09-25 추가 — North-up(heading-up 회전 전) 미리보기.

        get_map_image()는 항상 render_frame()이 만든 최종(heading-up 회전 완료)
        결과만 반환하므로, "route geometry 자체가 맞는지"(예: 경로가 보도 위에
        정상적으로 있는지)와 "heading-up 회전이 맞는지"를 지금까지는 분리해서 볼
        방법이 없었다. 이 메서드는 render_frame()이 warp 전에 그리는 것과 동일한
        past(회색)/future(빨강)/goal(주황) 선을 캐싱된 캔버스에 다시 그리기만 하고,
        **회전/리스케일/앵커 배치(진짜 heading-up 변환 수학)는 여기서 절대
        재구현하지 않는다** — 그건 항상 get_map_image()/render_frame()에서만 계산됨.
        heading_rad를 주면 참고용 화살표만 얹어서 그린다(디버그 전용 오버레이,
        모델 입력 지도에는 없음).
        """
        if self._route_latlon is None or self._canvas is None:
            return None
        out_size = out_size or self.out_size
        canvas_bgr, gx0, gy0 = self._canvas
        img = canvas_bgr.copy()
        idx, _ = self._closest_route_idx(lat, lon)
        route = self._route_latlon
        past_route, future_route = route[:idx + 1], route[idx:]

        def to_canvas_pts(seg):
            pts = []
            for la, lo in seg:
                gx, gy = latlon_to_pixel_global(la, lo, self.zoom)
                pts.append(global_to_canvas(gx, gy, gx0, gy0))
            return pts

        if len(past_route) >= 2:
            pts = to_canvas_pts(past_route)
            for k in range(1, len(pts)):
                cv2.line(img, pts[k - 1], pts[k], (160, 160, 160), 2, cv2.LINE_AA)

        ego_gx, ego_gy = latlon_to_pixel_global(future_route[0][0], future_route[0][1], self.zoom)
        ego_cx, ego_cy = global_to_canvas(ego_gx, ego_gy, gx0, gy0)
        if len(future_route) >= 1:
            pts = [(ego_cx, ego_cy)] + to_canvas_pts(future_route)
            for k in range(1, len(pts)):
                cv2.line(img, pts[k - 1], pts[k], (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(img, pts[-1], 5, (0, 165, 255), -1, cv2.LINE_AA)

        cv2.circle(img, (ego_cx, ego_cy), 6, (0, 200, 0), -1)
        cv2.circle(img, (ego_cx, ego_cy), 6, (255, 255, 255), 2)
        if heading_rad is not None:
            arrow_len_px = 40
            dx = math.cos(heading_rad) * arrow_len_px
            dy = -math.sin(heading_rad) * arrow_len_px  # East=0,North=+90 CCW -> screen dy는 반대 부호
            tip = (int(round(ego_cx + dx)), int(round(ego_cy + dy)))
            cv2.arrowedLine(img, (ego_cx, ego_cy), tip, (255, 0, 255), 3, cv2.LINE_AA, tipLength=0.35)

        rear_m = self.map_range_m * REAR_RATIO
        total_span_m = self.map_range_m + rear_m
        mpp = meters_per_pixel(lat, self.zoom)
        half_px = max(1, int((total_span_m / mpp) * 0.75))
        h, w = img.shape[:2]
        x1, x2 = max(0, ego_cx - half_px), min(w, ego_cx + half_px)
        y1, y2 = max(0, ego_cy - half_px), min(h, ego_cy + half_px)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    def get_map_image(self, lat, lon, heading_rad, past_track=None):
        """
        set_goal()이 먼저 호출되어 있어야 함. 네트워크 요청 없이 캐싱된 경로만 사용.
        past_track: 로봇이 실제로 지나온 (lat, lon) 리스트 (odometry/GPS 누적 기록,
                    최근 N개). None이면 과거 경로 없이(회색 선 없이) 렌더링.
        반환: (out_size, out_size, 3) uint8 RGB numpy array
        """
        if self._route_latlon is None or self._canvas is None:
            raise RuntimeError("set_goal(start_lat, start_lon, goal_lat, goal_lon)을 "
                                "먼저 호출해서 경로를 계산해야 합니다.")
        canvas_bgr, gx0, gy0 = self._canvas
        idx, _ = self._closest_route_idx(lat, lon)
        future_route = self._route_latlon[idx:]

        past_lats = past_lons = None
        if past_track:
            pt = np.array(past_track)
            # 2026-09-18(2차): ego와 같은 이유로 과거 궤적(회색)도 raw GPS 그대로 그리면
            # GPS 오차/드리프트로 실제 보도가 아니라 건물 위에 겹쳐 그려짐. 과거 점들도
            # 각각 캐싱된 경로 위 가장 가까운 점으로 스냅해서 전체 궤적(과거~현재~미래)이
            # 하나의 매끄러운 선으로 이어지게 한다. 네트워크 요청 없이 순수 벡터 연산.
            d = np.hypot(pt[:, [0]] - self._route_latlon[:, 0][None, :],
                         pt[:, [1]] - self._route_latlon[:, 1][None, :])
            snapped_past = self._route_latlon[np.argmin(d, axis=1)]
            past_lats, past_lons = snapped_past[:, 0], snapped_past[:, 1]

        # 2026-09-18: render_frame()에 raw GPS(lat,lon)를 그대로 ego 위치로 넘기면,
        # 로봇이 경로에서 몇 m만 벗어나 있어도(GPS 오차/드리프트로 흔함) "ego(벗어난 raw
        # 위치) → 경로 위 가장 가까운 점"을 잇는 첫 구간이 인위적으로 꺾여서 그려짐
        # (is_off_route 재라우팅 쿨다운을 넣어도, 로봇이 계속 그 근방에 있으면 매번
        # 똑같이 재현됨 — deploy_20260918_185324.jsonl). 대신 경로 위 가장 가까운 점
        # (future_route[0]과 동일한 점)을 ego 렌더링 위치로 써서 시작 구간을 항상
        # 경로와 정확히 일치시킨다. raw (lat,lon)은 is_off_route()/heading 추정 등
        # 실제 항법 로직에는 그대로 쓰이고, 여기 시각화용으로만 스냅됨.
        render_lat, render_lon = future_route[0]

        img = render_frame(
            canvas_bgr, gx0, gy0, self.zoom,
            render_lat, render_lon, heading_rad,
            future_route[:, 0], future_route[:, 1],
            past_lats, past_lons,
            out_size=self.out_size, map_range_m=self.map_range_m,
        )
        return img

    def draw_predicted_trajectory(self, map_img, pred_xy_m, target_step=2,
                                   color=(0, 255, 255)):
        """모델이 예측한 미래 waypoint(ego frame, x=forward m, y=left m, (N,2))를
        get_map_image()가 만든 ego-centric·heading-up 지도 위에 같은 스케일로 겹쳐 그림.

        지도가 heading-up이므로 ego 위치=이미지 정중앙, x=forward→위쪽(row 감소),
        y=left→왼쪽(col 감소). render_frame()과 동일한 px/m 스케일
        (out_size / (2*map_range_m))을 그대로 사용해야 실제 축척과 맞음.
        같이 그리는 흰 눈금선은 미터 단위 스케일바(축척 확인용).
        target_step: waypoint_to_control()이 실제로 조준하는 인덱스 — 더 크게 표시.
        """
        img = map_img.copy()
        rear_m = self.map_range_m * REAR_RATIO
        total_span_m = self.map_range_m + rear_m
        px_per_m = self.out_size / total_span_m
        cx = self.out_size / 2.0
        cy = (self.map_range_m / total_span_m) * self.out_size

        pts = []
        for x, y in pred_xy_m:
            col = int(round(cx - y * px_per_m))
            row = int(round(cy - x * px_per_m))
            pts.append((col, row))
        for k in range(1, len(pts)):
            cv2.line(img, pts[k - 1], pts[k], color, 2, cv2.LINE_AA)
        for k, p in enumerate(pts):
            r = 5 if k == target_step else 3
            cv2.circle(img, p, r, color, -1)
            cv2.circle(img, p, r, (255, 255, 255), 1)

        # 스케일바: map_range_m의 절반 정도 되는 "보기 좋은" 미터 값을 골라 하단에 표시
        bar_m = _pick_scale_bar_m(self.map_range_m)
        bar_px = int(round(bar_m * px_per_m))
        x0, y0 = 8, self.out_size - 10
        cv2.line(img, (x0, y0), (x0 + bar_px, y0), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(img, (x0, y0 - 4), (x0, y0 + 4), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(img, (x0 + bar_px, y0 - 4), (x0 + bar_px, y0 + 4), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"{bar_m:g}m", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def get_map_tensor(self, lat, lon, heading_rad, past_track=None, device="cpu"):
        """모델 forward()에 바로 넣을 수 있는 (1,3,96,96) 정규화된 텐서 반환."""
        img = self.get_map_image(lat, lon, heading_rad, past_track)
        pil = Image.fromarray(img)
        return self.transform(pil).unsqueeze(0).to(device)


if __name__ == "__main__":
    # 간단한 동작 확인 (OSRM 서버가 켜져 있어야 함)
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--heading_deg", type=float, default=0.0)
    p.add_argument("--goal_lat", type=float, required=True)
    p.add_argument("--goal_lon", type=float, required=True)
    p.add_argument("--map_range", type=float, default=20.0)
    p.add_argument("--out", type=str, default="/tmp/live_map_test.png")
    args = p.parse_args()

    builder = LiveMapBuilder(map_range_m=args.map_range)
    builder.set_goal(args.lat, args.lon, args.goal_lat, args.goal_lon)
    img = builder.get_map_image(args.lat, args.lon, math.radians(args.heading_deg))
    Image.fromarray(img).save(args.out)
    print(f"saved: {args.out}")
