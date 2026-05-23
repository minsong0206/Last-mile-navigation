# BEV Overview Map 설계 결정 사항

## 1. 개요

선별된 세그먼트별로 전체 경로를 한눈에 보여주는 BEV(Bird's Eye View) 조감도 이미지를 생성한다.
이 이미지는 학습 데이터 품질 검수 및 시각적 확인 용도로 사용된다.

---

## 2. 설계 결정 사항

### 2.1 시작점 중심 크롭 (Start-Centered Crop)

**결정**: 경로 전체 bbox 중심이 아닌, **세그먼트 시작점(GPS[0])을 크롭 중심**으로 사용한다.

**이유**:
- 로봇이 어디서 출발했는지 명확히 표시됨
- 미래 경로(빨간선)가 시작점 기준으로 뻗어 나가는 구조 → 진행 방향 직관적 파악
- bbox 중심 방식은 경로가 한쪽으로 치우칠 경우 시작점이 이미지 가장자리에 위치할 수 있음

**구현**:
- 시작점 픽셀 좌표 `(cx0, cy0)` 기준으로 `radius_px` 반경 정사각형 crop
- `radius_px = max(total_m * 1.5 / mpp, pad_px * 3)`
  - `total_m`: EKF 기준 총 이동거리 (미터)
  - `mpp`: meters_per_pixel (zoom, 위도 기반 계산)
  - 최소 크기 보장으로 매우 짧은 경로도 충분한 맥락 포함

---

### 2.2 거리 기반 BEV Zoom 자동 조정 (Adaptive Zoom)

**결정**: EKF 총 이동거리에 따라 BEV zoom 레벨을 자동으로 결정한다.

| 총 이동거리 | BEV Zoom | 픽셀당 미터 (위도 25° 기준) | 용도 |
|------------|---------|--------------------------|------|
| < 30m | base_zoom (18) | ~0.6 m/px | 매우 짧은 경로, 상세 확인 |
| 30 ~ 100m | base_zoom - 1 (17) | ~1.2 m/px | 일반 세그먼트 |
| > 100m | base_zoom - 2 (16) | ~2.4 m/px | 긴 경로, 전체 맥락 포함 |

**이유**:
- 짧은 경로(< 30m)를 낮은 zoom으로 보면 마커 2개만 보이고 경로가 점처럼 표시됨
- 긴 경로(> 100m)를 높은 zoom으로 보면 이미지가 너무 작아져 맥락 파악 불가
- 거리 비례 zoom으로 **어떤 세그먼트든 경로가 이미지의 30~70% 영역을 차지**하도록 조정

**ep6 결과 예시**:
```
seg00: total=71.9m → bev_zoom=17
seg01: total=71.4m → bev_zoom=17
seg02: total=18.7m → bev_zoom=18  ← 짧아서 zoom 유지
seg03: total=46.6m → bev_zoom=17
```

---

### 2.3 North-Up 고정 (No Rotation)

**결정**: BEV는 **north-up 고정**, 회전 없음.

**이유**:
- BEV는 학습 데이터가 아닌 **검수용 overview 이미지**이므로 지도 방향 일관성이 중요
- 회전 적용 시 이미지 코너에 흰색 여백이 생기고 직사각형 캔버스가 깨짐
- heading-up 정렬은 **프레임별 ego-centric 지도(osm_map_XXXXXX.png)** 에서 적용

---

### 2.4 시각 요소 (Visual Elements)

| 요소 | 색상 | 설명 |
|------|------|------|
| 경로선 | 주황색 `(204, 102, 0)` BGR | GPS 궤적 전체 |
| 시작점 마커 | 초록 `(0, 200, 0)` | 세그먼트 시작 위치 |
| 끝점 마커 | 빨강 `(0, 50, 230)` | 세그먼트 끝 위치 |
| 마커 테두리 | 흰색 `(255, 255, 255)` | 가독성 향상 |
| 배경 지도 | OSM tile (zoom-1 or zoom-2) | tile.openstreetmap.org |

---

## 3. 구현 함수 요약

```python
def _bev_zoom_for_distance(total_m, base_zoom):
    if total_m < 30:   return base_zoom
    elif total_m < 100: return max(base_zoom - 1, 15)
    else:              return max(base_zoom - 2, 15)

def save_bev(canvas_bgr, gx0, gy0, zoom, lats, lons, out_dir, fp=None, pad_px=80):
    # 1. EKF 거리 기반 zoom 결정
    # 2. 해당 zoom으로 타일 캔버스 빌드
    # 3. GPS 경로를 파란 선으로 오버레이
    # 4. 시작점/끝점 마커 표시
    # 5. 시작점 중심으로 radius_px 반경 crop
    # 6. PNG 저장
    return out_path, bev_zoom, total_m
```

---

## 4. 결론

| 항목 | 이전 방식 | 현재 방식 |
|------|-----------|-----------|
| 크롭 중심 | 경로 bbox 중심 | **시작점 중심** |
| Zoom | 항상 zoom-1 고정 | **거리 기반 자동 조정** |
| 회전 | 시도했다 제거 | **north-up 고정** |
| 짧은 경로 가시성 | 마커만 보임 | **zoom 유지로 상세 표시** |

시작점 중심 크롭과 거리 기반 zoom 조정을 결합하면,
짧은 세그먼트부터 긴 세그먼트까지 **일관된 가시성**으로 BEV를 생성할 수 있다.
