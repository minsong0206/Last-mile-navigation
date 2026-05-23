#!/usr/bin/env bash
# =============================================================================
# OSRM 전처리 스크립트 (extract → partition → customize)
# =============================================================================
#
# PBF 파일을 OSRM foot-profile로 전처리합니다.
# 전처리 완료 후 osrm/<region>/ 디렉토리에 .osrm 파일이 생성됩니다.
#
# 사용법:
#   bash scripts/preprocess_osrm.sh wuhan       # Wuhan만
#   bash scripts/preprocess_osrm.sh rides11     # output_rides_11 전체 (7개 지역)
#   bash scripts/preprocess_osrm.sh status      # PBF/OSRM 파일 상태 확인
#
# 지역별 PBF → 출력 디렉토리:
#   hubei-latest.osm.pbf         → osrm/wuhan/
#   philippines-latest.osm.pbf  → osrm/manila/
#   italy-latest.osm.pbf        → osrm/rome/
#   new-zealand-latest.osm.pbf  → osrm/wellington/
#   florida-latest.osm.pbf      → osrm/florida/
#   great-britain-latest.osm.pbf → osrm/brighton/
#   spain-latest.osm.pbf        → osrm/madrid/
#
# 소요 시간 (대략):
#   우한(Hubei)    ~10분
#   마닐라(PH)     ~20분
#   로마(Italy)    ~30분
#   웰링턴(NZ)     ~15분
#   플로리다(FL)   ~25분
#   브라이턴(GB)   ~60분  ← Great Britain은 대용량
#   마드리드(ES)   ~30분
# =============================================================================

OSRM_DIR="/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/osrm"
IMAGE="osrm/osrm-backend"
DOCKER_PROFILE="/usr/local/share/osrm/profiles/foot.lua"

preprocess() {
    local region=$1   # 출력 디렉토리 이름 (예: wuhan)
    local pbf=$2      # PBF 파일 이름 (예: hubei-latest.osm.pbf)
    local out_dir="$OSRM_DIR/$region"

    echo "========================================"
    echo "[$region] 전처리 시작"
    echo "  PBF : $OSRM_DIR/$pbf"
    echo "  출력: $out_dir"
    echo "========================================"

    # PBF 존재 확인
    if [[ ! -f "$OSRM_DIR/$pbf" ]]; then
        echo "  [$region] ERROR: PBF 파일 없음 — $OSRM_DIR/$pbf"
        return 1
    fi

    # 이미 전처리 완료 확인
    if [[ -f "$out_dir/$region.osrm" ]]; then
        echo "  [$region] 이미 전처리 완료 — 건너뜀"
        return 0
    fi

    mkdir -p "$out_dir"

    # PBF 복사 (Docker는 마운트 디렉토리 내부만 접근 가능)
    echo "  [$region] PBF 복사 중..."
    cp "$OSRM_DIR/$pbf" "$out_dir/$region.osm.pbf"

    echo "  [$region] Step 1/3: extract (foot profile)..."
    docker run --rm \
        -v "$out_dir:/data" \
        "$IMAGE" \
        osrm-extract -p "$DOCKER_PROFILE" "/data/$region.osm.pbf" \
        || { echo "  [$region] extract 실패"; return 1; }

    echo "  [$region] Step 2/3: partition..."
    docker run --rm \
        -v "$out_dir:/data" \
        "$IMAGE" \
        osrm-partition "/data/$region.osrm" \
        || { echo "  [$region] partition 실패"; return 1; }

    echo "  [$region] Step 3/3: customize..."
    docker run --rm \
        -v "$out_dir:/data" \
        "$IMAGE" \
        osrm-customize "/data/$region.osrm" \
        || { echo "  [$region] customize 실패"; return 1; }

    echo "  [$region] 전처리 완료 ✓"
}

status_check() {
    echo "PBF / OSRM 파일 상태:"
    echo ""
    printf "  %-12s  %-38s  %-6s  %-6s\n" "지역" "PBF 파일" "PBF" "OSRM"
    echo "  -----------------------------------------------------------------------"
    declare -A REGION_PBF=(
        [wuhan]="hubei-latest.osm.pbf"
        [manila]="philippines-latest.osm.pbf"
        [rome]="italy-latest.osm.pbf"
        [wellington]="new-zealand-latest.osm.pbf"
        [florida]="florida-latest.osm.pbf"
        [brighton]="great-britain-latest.osm.pbf"
        [madrid]="spain-latest.osm.pbf"
    )
    for region in wuhan manila rome wellington florida brighton madrid; do
        local pbf="${REGION_PBF[$region]}"
        local pbf_ok="❌" osrm_ok="❌"
        [[ -f "$OSRM_DIR/$pbf" ]]               && pbf_ok="✅"
        [[ -f "$OSRM_DIR/$region/$region.osrm" ]] && osrm_ok="✅"
        printf "  %-12s  %-38s  %-6s  %-6s\n" "$region" "$pbf" "$pbf_ok" "$osrm_ok"
    done
}

case "${1:-}" in
    wuhan)      preprocess wuhan      "hubei-latest.osm.pbf" ;;
    manila)     preprocess manila     "philippines-latest.osm.pbf" ;;
    rome)       preprocess rome       "italy-latest.osm.pbf" ;;
    wellington) preprocess wellington "new-zealand-latest.osm.pbf" ;;
    florida)    preprocess florida    "florida-latest.osm.pbf" ;;
    brighton)   preprocess brighton   "great-britain-latest.osm.pbf" ;;
    madrid)     preprocess madrid     "spain-latest.osm.pbf" ;;
    rides11)
        echo "output_rides_11 전체 지역 전처리 시작..."
        preprocess wuhan      "hubei-latest.osm.pbf"
        preprocess manila     "philippines-latest.osm.pbf"
        preprocess rome       "italy-latest.osm.pbf"
        preprocess wellington "new-zealand-latest.osm.pbf"
        preprocess florida    "florida-latest.osm.pbf"
        preprocess brighton   "great-britain-latest.osm.pbf"
        preprocess madrid     "spain-latest.osm.pbf"
        echo "모든 전처리 완료." ;;
    status)
        status_check ;;
    *)
        echo "사용법: $0 [wuhan|manila|rome|wellington|florida|brighton|madrid|rides11|status]"
        exit 1 ;;
esac
