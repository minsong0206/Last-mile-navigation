#!/usr/bin/env bash
# =============================================================================
# OSRM Docker 서버 시작 스크립트
# =============================================================================
#
# 10개 지역 foot-profile OSRM 서버를 Docker로 실행합니다:
#   Perth      → localhost:5001  (ride0 기존)
#   Taipei     → localhost:5002  (ride0 기존)
#   Tokyo      → localhost:5003  (ride0 기존)
#   Wuhan      → localhost:5004  (output_rides_11)
#   Manila     → localhost:5005  (output_rides_11)
#   Rome       → localhost:5006  (output_rides_11)
#   Wellington → localhost:5007  (output_rides_11)
#   Florida    → localhost:5008  (output_rides_11)
#   Brighton   → localhost:5009  (output_rides_11)
#   Madrid     → localhost:5010  (output_rides_11)
#
# 전처리된 OSRM 파일 위치:
#   osm_pipeline/osrm/<region>/
#
# 사용법:
#   bash scripts/start_osrm.sh              # 전체 시작
#   bash scripts/start_osrm.sh wuhan        # Wuhan만 시작
#   bash scripts/start_osrm.sh rides11      # output_rides_11 지역만 시작
#   bash scripts/start_osrm.sh status       # 실행 상태 확인
#   bash scripts/start_osrm.sh stop         # 전체 종료
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
OSRM_DIR="${OSRM_DIR:-$PIPELINE_DIR/osrm}"
IMAGE="osrm/osrm-backend"

start_server() {
    local region=$1
    local port=$2

    # 서브디렉토리(osrm/<region>/) 우선, 없으면 루트(osrm/)에서 직접 찾기
    local data_dir osrm_file
    if [[ -d "$OSRM_DIR/$region" ]]; then
        data_dir="$OSRM_DIR/$region"
        osrm_file=$(basename "$(ls "$data_dir"/*.osrm 2>/dev/null | head -1)")
    else
        data_dir="$OSRM_DIR"
        osrm_file=$(basename "$(ls "$data_dir/$region"*.osrm 2>/dev/null | head -1)")
    fi

    if [[ -z "$osrm_file" ]]; then
        echo "  [$region] ERROR: .osrm 파일을 찾을 수 없습니다 ($data_dir)"
        return 1
    fi

    # 이미 실행 중인지 확인
    if docker ps --format '{{.Names}}' | grep -q "osrm_$region"; then
        echo "  [$region] 이미 실행 중 (port $port)"
        return 0
    fi

    echo "  [$region] 시작 중... (port $port, file: $osrm_file)"
    docker run -d --rm \
        --name "osrm_$region" \
        -p "${port}:5000" \
        -v "$data_dir:/data" \
        "$IMAGE" \
        osrm-routed --algorithm mld "/data/$osrm_file" \
        > /dev/null

    # 준비 대기 (최대 15초)
    for i in $(seq 1 15); do
        if curl -sf "http://localhost:$port/route/v1/foot/0,0;0,0" > /dev/null 2>&1; then
            echo "  [$region] 준비 완료 (${i}s)"
            return 0
        fi
        sleep 1
    done
    echo "  [$region] WARNING: 15초 내 응답 없음 — 수동 확인 필요"
}

stop_server() {
    local region=$1
    if docker ps --format '{{.Names}}' | grep -q "osrm_$region"; then
        docker stop "osrm_$region" > /dev/null
        echo "  [$region] 종료됨"
    else
        echo "  [$region] 실행 중 아님"
    fi
}

status_check() {
    echo "OSRM 서버 상태:"
    local regions=("perth:5001" "taipei:5002" "tokyo:5003" "wuhan:5004" "manila:5005" "rome:5006" "wellington:5007" "florida:5008" "brighton:5009" "madrid:5010")
    for entry in "${regions[@]}"; do
        local region="${entry%%:*}"
        local port="${entry##*:}"
        if docker ps --format '{{.Names}}' | grep -q "osrm_$region"; then
            echo "  [$region] 실행 중 (port $port)"
        else
            echo "  [$region] 중지됨"
        fi
    done
}

case "${1:-all}" in
    perth)      start_server perth      5001 ;;
    taipei)     start_server taipei     5002 ;;
    tokyo)      start_server tokyo      5003 ;;
    wuhan)      start_server wuhan      5004 ;;
    manila)     start_server manila     5005 ;;
    rome)       start_server rome       5006 ;;
    wellington) start_server wellington 5007 ;;
    florida)    start_server florida    5008 ;;
    brighton)   start_server brighton   5009 ;;
    madrid)     start_server madrid     5010 ;;
    rides11)
        echo "output_rides_11 OSRM 서버 시작 (ports 5004-5010)..."
        start_server wuhan      5004
        start_server manila     5005
        start_server rome       5006
        start_server wellington 5007
        start_server florida    5008
        start_server brighton   5009
        start_server madrid     5010
        echo "완료."
        status_check ;;
    stop)
        echo "모든 OSRM 서버 종료..."
        for region in perth taipei tokyo wuhan manila rome wellington florida brighton madrid; do
            stop_server "$region"
        done ;;
    status)
        status_check ;;
    all)
        echo "모든 OSRM 서버 시작..."
        start_server perth      5001
        start_server taipei     5002
        start_server tokyo      5003
        start_server wuhan      5004
        start_server manila     5005
        start_server rome       5006
        start_server wellington 5007
        start_server florida    5008
        start_server brighton   5009
        start_server madrid     5010
        echo "완료."
        status_check ;;
    *)
        echo "사용법: $0 [perth|taipei|tokyo|wuhan|manila|rome|wellington|florida|brighton|madrid|rides11|all|stop|status]"
        exit 1 ;;
esac
