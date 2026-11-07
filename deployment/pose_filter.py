"""상보필터(complementary filter) 기반 pose(위치+heading) 추정 — 배포용 (2026-11-07).

배경: Test 2/2b(2026-11-07) 실주행에서 "지도 위치/각도가 로봇 현실과 어긋나 전진·회전
실패"가 반복됨. 원인은 배포가 로봇 pose를 제대로 추정하지 않고 (a) heading을 경로
첫 세그먼트에서 seed만 하고 교정 안 함(gyro_heading_gps_gain=0), (b) 위치는 GPS
동결 중 그대로 멈춤 에 의존한 것.

이 필터는 2026-11-01 오프라인 평가에서 최적이었던 B2_comp(상보필터)를 배포에 맞게
포팅한 것. imu_gps_fusion_replay.py의 B2Comp와 동일한 수식:
  - 예측: 바퀴속도 v로 전진, (자이로 z, 없으면 rpm 좌우차)로 회전
  - 보정: 새 GPS마다 위치를 β만큼 GPS로 당기고, GPS 진행방향(누적 1.5m)으로 heading을 α만큼 당김
평가 기준 최적 파라미터: W=0.4, gscale=π/180(자이로 deg/s→rad/s), α=0.3, β=0.9.

좌표/부호 규약 (배포 코드와 동일):
  - heading_rad = atan2(dlat_north, dlon_east), 즉 East=0, 북=+90°, CCW 양수(수학각).
    route_bearing_rad()/estimate_heading_from_track()과 동일.
  - 자이로 z 양수 = 좌회전(CCW) = heading 증가.
  - ENU: x=동(m), y=북(m).

배포와의 차이(개선): 콜드 init(3m 이동 후 course로 heading 초기화) 대신, 경로 첫
세그먼트 방향으로 seed()해서 GO LIVE 즉시 heading이 있고(기존 gyro와 동일 전제:
"로봇을 경로 방향으로 놓는다"), 그 뒤 GPS course로 α만큼 지속 교정한다 — 즉 틀린
seed가 주행 중 스스로 보정된다(Test 2b에서 빠졌던 부분).

순수 계산 모듈. 로봇/네트워크를 건드리지 않음. math 표준 라이브러리만 사용.
"""
import math

DEFAULT_TRACK_WIDTH_M = 0.4          # 스키드 스티어 유효 윤거 W
DEFAULT_GYRO_SCALE = math.pi / 180.0  # 자이로 z(deg/s) → rad/s
DEFAULT_ALPHA = 0.3                   # heading GPS-course 보정 계수
DEFAULT_BETA = 0.9                    # 위치 GPS 보정 계수
COURSE_MIN_DISP_M = 1.5              # heading 보정을 켜는 최소 GPS 누적 변위

# 근거리 평면 근사(배포 전역과 동일 상수)
_M_PER_DEG_LAT = 110540.0
def _m_per_deg_lon(lat):
    return 111320.0 * math.cos(math.radians(lat))


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class PoseFilter:
    def __init__(self, track_width_m=DEFAULT_TRACK_WIDTH_M, gyro_scale=DEFAULT_GYRO_SCALE,
                 alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, course_min_disp_m=COURSE_MIN_DISP_M):
        self.W = track_width_m
        self.gscale = gyro_scale
        self.alpha = alpha
        self.beta = beta
        self.course_min_disp_m = course_min_disp_m
        # 상태
        self.lat0 = self.lon0 = None     # ENU 원점(첫 seed/GPS)
        self.x = self.y = None           # ENU 위치(m)
        self.th = None                   # heading(rad, East=0 CCW)
        self._course_anchor = None       # heading 보정용 앵커(ENU)
        self.initialized = False

    # ── 좌표 변환 ──
    def _to_enu(self, lat, lon):
        return ((lon - self.lon0) * _m_per_deg_lon(self.lat0),
                (lat - self.lat0) * _M_PER_DEG_LAT)

    def _to_latlon(self, x, y):
        return (self.lat0 + y / _M_PER_DEG_LAT,
                self.lon0 + x / _m_per_deg_lon(self.lat0))

    # ── 초기화 ──
    def seed(self, lat, lon, heading_rad):
        """GO LIVE 시점: 현재 GPS를 원점/위치로, 경로 첫 세그먼트 방향을 초기 heading으로."""
        self.lat0, self.lon0 = lat, lon
        self.x, self.y = 0.0, 0.0
        self.th = _wrap(heading_rad)
        self._course_anchor = (0.0, 0.0)
        self.initialized = True

    # ── 매 틱 갱신 ──
    def update(self, dt_s, wheel_speed_mps, gyro_z_deg_s, moving, gps_latlon):
        """한 틱 진행. 반환 (lat, lon, heading_rad). seed() 전이면 (None,None,None).

        dt_s           : 직전 틱과의 시간차(초, 2초로 클램프 권장)
        wheel_speed_mps: 바퀴 선속도(m/s) — wheel_speed_mps(raw) 환산값
        gyro_z_deg_s   : 자이로 z 평균(deg/s), 없으면 None → rpm 좌우차로 대체 못하면 0
        moving         : 로봇 이동 여부(정지면 회전/전진 0)
        gps_latlon     : 이번 틱에 새로 들어온 GPS (lat,lon) 또는 None(동결/미갱신)
        """
        if not self.initialized:
            # 아직 seed 안 됨 — GPS가 있으면 원점만 잡아둠(heading은 seed()로 받음)
            if gps_latlon is not None and self.lat0 is None:
                self.lat0, self.lon0 = gps_latlon
            return (None, None, None)

        dt = max(0.0, min(dt_s, 2.0))
        # ── 예측(회전 → 전진) ──
        if moving:
            w = (gyro_z_deg_s * self.gscale) if gyro_z_deg_s is not None else 0.0
            self.th = _wrap(self.th + w * dt)
            v = wheel_speed_mps
            self.x += v * math.cos(self.th) * dt
            self.y += v * math.sin(self.th) * dt

        # ── 보정(새 GPS가 있을 때만) ──
        if gps_latlon is not None:
            gx, gy = self._to_enu(*gps_latlon)
            # 위치: β만큼 GPS로 당김
            self.x += self.beta * (gx - self.x)
            self.y += self.beta * (gy - self.y)
            # heading: 앵커에서 1.5m 이상 벌어졌으면 그 course로 α만큼 당김
            dx, dy = gx - self._course_anchor[0], gy - self._course_anchor[1]
            if math.hypot(dx, dy) >= self.course_min_disp_m:
                course = math.atan2(dy, dx)
                self.th = _wrap(self.th + self.alpha * _wrap(course - self.th))
                self._course_anchor = (gx, gy)

        lat, lon = self._to_latlon(self.x, self.y)
        return (lat, lon, self.th)


def gyro_z_deg_s(raw_data):
    """frodobot_raw의 gyros 버스트에서 z축(yaw) 평균(deg/s). 없으면 None.
    (2026-11-01 Phase B에서 자이로 z 단위 deg/s 확정, 양수=좌회전.)"""
    if not raw_data:
        return None
    gs = raw_data.get("gyros") or []
    zs = [g[2] for g in gs if isinstance(g, (list, tuple)) and len(g) >= 3]
    return sum(zs) / len(zs) if zs else None
