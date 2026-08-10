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
  - MAP_RANGE_M, 회전 공식(rot_deg = 90 - heading_deg), heading-up 정렬
  - ROUTE_COLOR(빨강)=미래 경로, PAST_COLOR(회색)=과거 경로, ROUTE_WIDTH
  - 최종 리사이즈 96x96 + IMG_MEAN/IMG_STD 정규화 (모델 입력 직전 rides11_dataset과 동일)

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

  # 경로 이탈이 심하면(예: 15m 이상) 재계산:
  if builder.is_off_route(lat, lon, threshold_m=15.0):
      builder.set_goal(lat, lon, goal_lat, goal_lon)
"""

import sys
import math
from pathlib import Path

import numpy as np
import requests
import torch
from torchvision import transforms
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "osm_pipeline" / "py"))
from osm_map_generator import (
    osrm_port, build_canvas, render_frame, _densify_route,
    MAP_RANGE_M as TRAIN_MAP_RANGE_M, ZOOM as TRAIN_ZOOM, MAP_SIZE_PX,
)

# rides11_dataset.py와 동일한 정규화 (모델이 학습 때 본 것과 동일한 분포로 맞추기 위함)
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

LAT_M = 111320.0  # 위도 1도당 미터 (haversine 근사 대신 간단한 등적원통 근사, 근거리 오차 무시 가능)


def query_osrm_route(start_lat, start_lon, goal_lat, goal_lon, port,
                      interp_step_m=1.0, timeout=3.0, max_retries=3, retry_delay=1.0):
    """출발→도착 OSRM foot 경로 쿼리. 배포 시작 시 1번만 호출됨 (osmnav의
    navigation_node.py 패턴 참고 — retry 포함). 실패 시 직선 경로로 폴백.
    반환: (M,2) [[lat,lon], ...], interp_step_m 간격으로 densify됨."""
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
            return _densify_route(route, interp_step_m)
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
    print(f"[build_live_map] OSRM 쿼리 {max_retries}회 실패({last_exc}), 직선 경로로 폴백")
    route = np.array([[start_lat, start_lon], [goal_lat, goal_lon]])
    return _densify_route(route, interp_step_m)


class LiveMapBuilder:
    def __init__(self, map_range_m: float = TRAIN_MAP_RANGE_M, zoom: int = TRAIN_ZOOM,
                 out_size: int = MAP_SIZE_PX, model_input_size: int = 96):
        """
        map_range_m: 학습 때 쓴 것과 반드시 동일해야 함 (예: 20m 재학습 체크포인트를
                     쓴다면 여기도 20.0으로 맞출 것 — 다르면 학습-추론 분포 불일치).
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

    def set_goal(self, start_lat, start_lon, goal_lat, goal_lon):
        """경로를 1번 계산해서 캐싱. 배포 시작 시(또는 경로 이탈 재계산 시)만 호출.
        get_map_image()는 이후 이 캐싱된 경로만 사용하고 네트워크 요청을 하지 않음."""
        port = osrm_port(start_lat, start_lon)
        self._route_latlon = query_osrm_route(start_lat, start_lon, goal_lat, goal_lon, port)
        self._canvas = build_canvas(self._route_latlon[:, 0], self._route_latlon[:, 1],
                                     self.zoom, self.session)
        print(f"[LiveMapBuilder] 경로 캐싱 완료: {len(self._route_latlon)}개 포인트 "
              f"(출발=({start_lat:.5f},{start_lon:.5f}) → 목표=({goal_lat:.5f},{goal_lon:.5f}))")

    def _closest_route_idx(self, lat, lon):
        """process_episode()의 map_frames_to_route()와 같은 발상: 현재 위치가
        캐싱된 경로의 어느 지점에 가장 가까운지 찾는다 (매 프레임 O(M), M은
        보통 수백 개 이내라 실시간 루프에 문제없음)."""
        d = np.hypot(self._route_latlon[:, 0] - lat, self._route_latlon[:, 1] - lon)
        return int(np.argmin(d)), float(d.min())

    def is_off_route(self, lat, lon, threshold_m=15.0):
        """경로에서 threshold_m 이상 벗어났는지 확인 (osmnav의 재라우팅 트리거와 동일 개념).
        True면 호출 측(배포 루프)에서 set_goal()을 다시 불러 재계산해야 함."""
        if self._route_latlon is None:
            return False
        _, dist_deg = self._closest_route_idx(lat, lon)
        dist_m = dist_deg * LAT_M  # 위경도 차 → 대략적인 미터 환산 (근거리 근사)
        return dist_m > threshold_m

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
            past_lats, past_lons = pt[:, 0], pt[:, 1]

        img = render_frame(
            canvas_bgr, gx0, gy0, self.zoom,
            lat, lon, heading_rad,
            future_route[:, 0], future_route[:, 1],
            past_lats, past_lons,
            out_size=self.out_size, map_range_m=self.map_range_m,
        )
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
