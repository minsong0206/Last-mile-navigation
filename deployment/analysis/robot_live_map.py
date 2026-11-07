"""로봇 위치 실시간 지도 — 브라우저에서 로봇이 지금 어디 있는지(보도 위인지) 실시간으로 본다.

작은 로컬 웹서버. 브라우저로 http://127.0.0.1:8090 를 열면 Leaflet + OSM 지도(보도 선 보임)
위에 로봇 현재 위치가 1.5초마다 갱신되며 찍힌다. 가장 가까운 보도까지의 거리(OSRM foot
nearest)도 실시간으로 숫자+원으로 표시 — ≤3m면 초록(보도 위), 멀면 빨강.

브라우저가 OSM 타일을 직접 받으므로(서버 스크래핑 아님) 타일 API 키 문제 없음.
로봇을 움직이지 않음(읽기 전용: SDK /data, OSRM nearest만 조회).

실행:
  python3 deployment/analysis/robot_live_map.py
  그다음 브라우저에서:  http://127.0.0.1:8090
  (터미널에서 바로:  xdg-open http://127.0.0.1:8090 )

--goal LAT LON 을 주면 목표도 지도에 핀으로 표시.
"""
import argparse
import json
import math
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SDK = "http://127.0.0.1:8000"
OSRM = "http://127.0.0.1:5011"
PORT = 8090
GOAL = None

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>로봇 실시간 위치</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body{margin:0;height:100%;font-family:system-ui,sans-serif}
  #map{position:absolute;inset:0}
  #hud{position:absolute;z-index:1000;top:10px;left:10px;background:#fff;padding:10px 13px;
    border-radius:8px;box-shadow:0 1px 6px rgba(0,0,0,.3);font-size:14px;line-height:1.5;max-width:260px}
  #hud b{font-size:15px}
  .ok{color:#1f8a4c}.bad{color:#c0392b}.warn{color:#b5730b}
</style></head><body>
<div id="map"></div>
<div id="hud">연결 중…</div>
<script>
const map = L.map('map').setView([37.63,127.075], 19);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom:20, attribution:'© OpenStreetMap'}).addTo(map);
let robot=null, acc=null, snapLine=null, goalMk=null, routeLine=null, routeStart=null, first=true;
const GOAL = __GOAL__;
if (GOAL) L.marker(GOAL,{title:'목표'}).addTo(map).bindPopup('목표');
async function tick(){
  try{
    const r = await fetch('/pos'); const d = await r.json();
    if(d.error){ document.getElementById('hud').innerHTML='<b class=bad>오류</b><br>'+d.error; return; }
    const ll=[d.lat,d.lon];
    if(!robot){
      robot=L.circleMarker(ll,{radius:8,color:'#1565c0',fillColor:'#1565c0',fillOpacity:1}).addTo(map);
    } else robot.setLatLng(ll);
    if(first){ map.setView(ll,19); first=false; }
    // 가장 가까운 보도 점까지 선
    if(d.snap_lat!=null){
      if(snapLine) snapLine.remove();
      const col = d.snap_m<=3?'#1f8a4c':(d.snap_m<10?'#b5730b':'#c0392b');
      snapLine=L.polyline([ll,[d.snap_lat,d.snap_lon]],{color:col,weight:3,dashArray:'4'}).addTo(map);
    }
    // OSRM 경로(로봇 현위치→목표) — 로봇이 움직일 때마다 다시 그림
    let routeTxt='';
    if(d.route && d.route.length>1){
      if(routeLine) routeLine.remove();
      if(routeStart) routeStart.remove();
      routeLine=L.polyline(d.route,{color:'#e8468c',weight:4,opacity:.85}).addTo(map);
      // 경로 첫 세그먼트 방향(로봇을 세울 방향) 강조
      routeStart=L.polyline([d.route[0],d.route[1]],{color:'#000',weight:6,opacity:.9}).addTo(map);
      routeTxt='<br>경로 '+d.route_dist_m.toFixed(0)+'m, 총회전 '+
        (d.route_turn_deg>0?'+':'')+d.route_turn_deg.toFixed(0)+'° '+
        '('+(d.route_turn_deg<-10?'좌회전':d.route_turn_deg>10?'우회전':'직진')+')'+
        '<br>첫 세그먼트 '+d.route_first_deg.toFixed(0)+'° '+
        (d.route_backward?'<b class=bad>⚠반대로 꺾임</b>':'<span class=ok>정상</span>');
    }
    const cls = d.snap_m<=3?'ok':(d.snap_m<10?'warn':'bad');
    const mark = d.snap_m<=3?'✅ 보도 위':(d.snap_m<10?'⚠ 가까움':'❌ 멀다');
    document.getElementById('hud').innerHTML =
      '<b>로봇 위치</b><br>'+d.lat.toFixed(6)+', '+d.lon.toFixed(6)+
      '<br>가장 가까운 보도: <b class='+cls+'>'+d.snap_m.toFixed(1)+'m '+mark+'</b>'+
      '<br>GPS fix='+d.fix_quality+' (2=양호)'+routeTxt+
      '<br><span style="color:#e8468c">━</span> 경로 <span style="color:#000">━</span> 첫 세그먼트';
  }catch(e){ document.getElementById('hud').innerHTML='<b class=bad>서버 연결 끊김</b>'; }
}
tick(); setInterval(tick,1500);
</script></body></html>"""


def _get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def _bearing(a, b):
    dlat = b[0] - a[0]
    dE = (b[1] - a[1]) * math.cos(math.radians(a[0]))
    return math.degrees(math.atan2(dE, dlat)) % 360


def _hav(a, b):
    R = 6371000.0
    dlat = math.radians(b[0] - a[0]); dlon = math.radians(b[1] - a[1])
    x = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(a[0])) * math.cos(math.radians(b[0])) * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(x))


def pos_json():
    d = _get(f"{SDK}/data", timeout=30)
    lat, lon = float(d["latitude"]), float(d["longitude"])
    out = {"lat": lat, "lon": lon, "fix_quality": d.get("fix_quality"),
           "snap_m": None, "snap_lat": None, "snap_lon": None}
    try:
        nr = _get(f"{OSRM}/nearest/v1/foot/{lon},{lat}?number=1", timeout=5)
        if nr.get("code") == "Ok" and nr.get("waypoints"):
            w = nr["waypoints"][0]
            out["snap_m"] = w["distance"]
            out["snap_lat"], out["snap_lon"] = w["location"][1], w["location"][0]
    except Exception:
        pass
    # 목표가 있으면 로봇 현위치→목표 OSRM 경로도 매 틱 새로 뽑아 보냄
    if GOAL is not None:
        try:
            url = (f"{OSRM}/route/v1/foot/{lon},{lat};{GOAL[1]},{GOAL[0]}"
                   f"?overview=full&geometries=geojson&steps=false")
            rt = _get(url, timeout=5)
            if rt.get("code") == "Ok" and rt.get("routes"):
                r0 = rt["routes"][0]
                coords = [(c[1], c[0]) for c in r0["geometry"]["coordinates"]]
                out["route"] = coords
                out["route_dist_m"] = r0["distance"]
                brs = [_bearing(coords[i], coords[i + 1]) for i in range(len(coords) - 1)
                       if _hav(coords[i], coords[i + 1]) > 0.3]

                def ad(a, b):
                    return (b - a + 180) % 360 - 180
                out["route_turn_deg"] = (
                    sum(ad(brs[i - 1], brs[i]) for i in range(1, len(brs))) if len(brs) > 1 else 0.0)
                first = brs[0] if brs else _bearing(coords[0], (GOAL[0], GOAL[1]))
                straight = _bearing(coords[0], (GOAL[0], GOAL[1]))
                out["route_first_deg"] = first
                out["route_backward"] = abs(ad(first, straight)) > 60
        except Exception:
            pass
    return out


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            goal_js = ("[%f,%f]" % (GOAL[0], GOAL[1])) if GOAL else "null"
            html = PAGE.replace("__GOAL__", goal_js).encode("utf-8")
            self._send(200, html, "text/html; charset=utf-8")
        elif self.path == "/pos":
            try:
                body = json.dumps(pos_json()).encode()
            except Exception as e:
                body = json.dumps({"error": str(e)}).encode()
            self._send(200, body, "application/json")
        else:
            self._send(404, b"not found", "text/plain")


def main():
    global GOAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", nargs=2, type=float, metavar=("LAT", "LON"), default=None)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    GOAL = args.goal
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    print(f"[robot_live_map] 브라우저에서 열기:  http://127.0.0.1:{args.port}")
    print(f"  (터미널에서 바로:  xdg-open http://127.0.0.1:{args.port} )")
    print("  Ctrl+C로 종료.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[종료]")


if __name__ == "__main__":
    main()
