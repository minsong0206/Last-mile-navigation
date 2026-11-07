"""로봇 배치 확인 — "지금 로봇이 지도에 그려진 보도 위에 있나?"를 출발 전에 본다 (2026-11-07).

Test 2/2b 실패의 한 축은 로봇이 OSM 보도에서 멀리(실측 12.5m) 떨어진 데서 출발해
지도 ego 위치가 엉뚱한 점에 스냅·고착된 것. 이 스크립트는 SDK(/data)에서 로봇의
현재 GPS를 읽어, OSRM foot 네트워크상 가장 가까운 보도까지의 거리(=배포 때 start_snap)와
지도를 눈으로 볼 수 있는 OpenStreetMap 링크를 출력한다. --watch로 반복 출력하면
로봇을 옮기며 snap 거리가 줄어드는 걸 실시간으로 볼 수 있다.

--goal 를 주면 그 목표까지 OSRM 경로도 뽑아 ① 첫 세그먼트 방위(= 로봇을 세울 방향)
② 반대-스텁 여부 ③ 경로의 총 회전량까지 알려준다.

로봇을 움직이지 않음(읽기 전용). SDK(8000)·OSRM(5011) 서버가 떠 있어야 함.

실행:
  python3 deployment/analysis/check_robot_placement.py
  python3 deployment/analysis/check_robot_placement.py --watch
  python3 deployment/analysis/check_robot_placement.py --goal 37.6291 127.0748
"""
import argparse
import json
import math
import time
import urllib.request

SDK = "http://127.0.0.1:8000"
OSRM = "http://127.0.0.1:5011"
SNAP_OK_M = 3.0      # 이 거리 이내면 "보도 위"로 간주


def _get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def robot_gps():
    d = _get(f"{SDK}/data", timeout=30)
    return float(d["latitude"]), float(d["longitude"]), d


def osrm_nearest(lat, lon):
    """가장 가까운 foot 네트워크 점까지의 거리(m)와 그 좌표."""
    d = _get(f"{OSRM}/nearest/v1/foot/{lon},{lat}?number=1", timeout=10)
    if d.get("code") != "Ok" or not d.get("waypoints"):
        return None
    w = d["waypoints"][0]
    return w["distance"], (w["location"][1], w["location"][0])  # (dist_m, (lat,lon))


def comp_bearing(a, b):
    dlat = b[0] - a[0]
    dE = (b[1] - a[1]) * math.cos(math.radians(a[0]))
    return math.degrees(math.atan2(dE, dlat)) % 360


def haversine(a, b):
    R = 6371000.0
    dlat = math.radians(b[0] - a[0]); dlon = math.radians(b[1] - a[1])
    x = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(a[0])) * math.cos(math.radians(b[0])) * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(x))


def compass_name(deg):
    names = ["북", "북북동", "북동", "동북동", "동", "동남동", "남동", "남남동",
             "남", "남남서", "남서", "서남서", "서", "서북서", "북서", "북북서"]
    return names[int((deg % 360) / 22.5 + 0.5) % 16]


def route_info(slat, slon, glat, glon):
    url = (f"{OSRM}/route/v1/foot/{slon},{slat};{glon},{glat}"
           f"?overview=full&geometries=geojson&steps=false")
    d = _get(url, timeout=10)
    if d.get("code") != "Ok":
        return None
    rt = d["routes"][0]
    coords = [(c[1], c[0]) for c in rt["geometry"]["coordinates"]]
    # 유효 세그먼트 방위들(>0.3m)
    brs = [comp_bearing(coords[i], coords[i + 1]) for i in range(len(coords) - 1)
           if haversine(coords[i], coords[i + 1]) > 0.3]
    def ad(a, b):
        return (b - a + 180) % 360 - 180
    cum = sum(ad(brs[i - 1], brs[i]) for i in range(1, len(brs))) if len(brs) > 1 else 0.0
    straight = comp_bearing(coords[0], (glat, glon))
    first = brs[0] if brs else straight
    # 첫 10m 방향
    far = next((c for c in coords if haversine(coords[0], c) >= 10), coords[-1])
    first10 = comp_bearing(coords[0], far)
    backward = abs(ad(first, straight)) > 60
    return {
        "dist_m": rt["distance"], "n_pts": len(coords),
        "first_seg_deg": first, "first10_deg": first10,
        "straight_deg": straight, "total_turn_deg": cum, "backward_stub": backward,
    }


def report(goal=None):
    rlat, rlon, raw = robot_gps()
    near = osrm_nearest(rlat, rlon)
    print("=" * 64)
    print(f"로봇 현재 GPS = {rlat:.7f}, {rlon:.7f}  (fix_quality={raw.get('fix_quality')}, "
          f"hdop={raw.get('hdop'):.4f})")
    osm = f"https://www.openstreetmap.org/?mlat={rlat}&mlon={rlon}#map=19/{rlat}/{rlon}"
    print(f"지도에서 보기(보도 표시됨): {osm}")
    if near:
        dist, loc = near
        mark = "✅ 보도 위" if dist <= SNAP_OK_M else ("⚠ 보도에서 멀다" if dist < 10 else "❌ 보도에서 많이 떨어짐")
        print(f"가장 가까운 보도까지: {dist:.1f} m  {mark}  (기준 ≤{SNAP_OK_M}m)")
        if dist > SNAP_OK_M:
            b = comp_bearing((rlat, rlon), loc)
            print(f"  → 보도는 {compass_name(b)}({b:.0f}°) 방향 {dist:.1f}m. 로봇을 그쪽으로 옮겨 다시 확인.")
    else:
        print("가장 가까운 보도: OSRM nearest 실패(서버/좌표 확인)")
    if goal:
        glat, glon = goal
        ri = route_info(rlat, rlon, glat, glon)
        print("-" * 64)
        if ri is None:
            print("목표까지 OSRM 경로 실패.")
        else:
            print(f"목표까지 경로: {ri['dist_m']:.1f} m, 점 {ri['n_pts']}개, "
                  f"총 회전 {ri['total_turn_deg']:+.0f}° "
                  f"({'좌회전' if ri['total_turn_deg']<-10 else '우회전' if ri['total_turn_deg']>10 else '거의 직선'})")
            print(f"경로 첫 세그먼트 = {ri['first_seg_deg']:.0f}°({compass_name(ri['first_seg_deg'])}), "
                  f"첫 10m = {ri['first10_deg']:.0f}°")
            if ri["backward_stub"]:
                print(f"  ❌ 반대-스텁 경고: 첫 세그먼트({ri['first_seg_deg']:.0f}°)가 목표 직선방향"
                      f"({ri['straight_deg']:.0f}°)과 60°+ 어긋남 — 이 자리에서 출발하면 Test 2처럼 꺾임. 위치 옮기기 권장.")
            else:
                print(f"  ✅ 경로 시작 정상. **로봇을 {compass_name(ri['first10_deg'])}"
                      f"({ri['first10_deg']:.0f}°, 경로 첫 10m 방향)을 바라보게** 세우세요.")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", nargs=2, type=float, metavar=("LAT", "LON"), default=None)
    ap.add_argument("--watch", action="store_true", help="2초마다 반복(로봇 옮기며 확인)")
    args = ap.parse_args()
    if args.watch:
        try:
            while True:
                report(args.goal)
                time.sleep(2.0)
        except KeyboardInterrupt:
            print("\n[종료]")
    else:
        report(args.goal)


if __name__ == "__main__":
    main()
