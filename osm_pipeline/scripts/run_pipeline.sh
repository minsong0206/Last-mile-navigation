#!/usr/bin/env bash
# =============================================================================
# FrodoBots OSM Navigation Pipeline
# =============================================================================
#
# 전체 파이프라인 3단계:
#   Step 1. Episode Selection  — OSRM 경로와 EKF 궤적 비교로 유효 세그먼트 선별
#   Step 2. OSM Map Generation — 선별된 세그먼트의 프레임별 ego-centric 지도 생성
#              + GPS CSV 저장
#              + BEV 조감도 저장
#
# 사전 요건:
#   - conda 환경 'mbra' 활성화 (pyarrow, utm, scipy, requests, opencv, pillow)
#   - OSRM Docker 서버 3개 실행 중:
#       Perth  → localhost:5001
#       Taipei → localhost:5002
#       Tokyo  → localhost:5003
#
# 사용법:
#   bash scripts/run_pipeline.sh                  # 전체 실행
#   bash scripts/run_pipeline.sh --ep 6           # 특정 에피소드만
#   bash scripts/run_pipeline.sh --ep 6 --seg 1   # 특정 세그먼트만
#   bash scripts/run_pipeline.sh --skip-selection # 선별 스킵 (이미 완료된 경우)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
CONDA_ENV="mbra"

EP=""
SEG=""
SKIP_SELECTION=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ep)           EP="$2";  shift 2 ;;
        --seg)          SEG="$2"; shift 2 ;;
        --skip-selection) SKIP_SELECTION=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

run_python() {
    conda run -n "$CONDA_ENV" python3 "$@"
}

cd "$PIPELINE_DIR"

echo "========================================================"
echo "  FrodoBots OSM Navigation Pipeline"
echo "  작업 디렉토리: $PIPELINE_DIR"
echo "========================================================"

# ── Step 1: Episode Selection ─────────────────────────────────────────────────
if [[ $SKIP_SELECTION -eq 0 && -z "$EP" ]]; then
    echo ""
    echo "[Step 1] Episode Selection 실행 중..."
    echo "  - 정적 프레임 제거 (EKF 이동거리 < 5cm)"
    echo "  - 긴 정지 구간(≥50 프레임)으로 세그먼트 분리"
    echo "  - 보행자 도로 snap 거리 확인 (OSRM foot profile, ≤15m)"
    echo "  - Fréchet / Chamfer / Heading / 길이비율 임계값 필터링"
    echo ""
    run_python episode_selector.py
    echo "  → episode_scores.json 저장 완료"
else
    echo "[Step 1] Skipped (--skip-selection 또는 --ep 지정됨)"
fi

# ── Step 2: OSM Map Generation ────────────────────────────────────────────────
echo ""
echo "[Step 2] OSM Map Generation 실행 중..."
echo "  - zoom=18 OSM 타일 + OSRM foot-profile 경로"
echo "  - 224×224 ego-centric, heading-up PNG (프레임별)"
echo "  - gps.csv 저장 (frame_index, latitude, longitude)"
echo "  - bev_overview.png 저장 (zoom=17 전체 경로 조감도)"
echo ""

MAP_ARGS=""
if [[ -n "$EP" && -n "$SEG" ]]; then
    MAP_ARGS="--ep $EP --seg $SEG"
    echo "  대상: ep${EP} seg${SEG}"
elif [[ -n "$EP" ]]; then
    MAP_ARGS="--ep $EP"
    echo "  대상: ep${EP} 전체 세그먼트"
else
    echo "  대상: 선별된 전체 세그먼트"
fi

run_python osm_map_generator.py $MAP_ARGS

echo ""
echo "========================================================"
echo "  파이프라인 완료"
echo "  출력 위치: $PIPELINE_DIR/osm_maps/"
echo "  각 폴더 구성:"
echo "    osm_map_XXXXXX.png  — 프레임별 ego-centric 지도"
echo "    gps.csv             — 프레임별 GPS 좌표"
echo "    bev_overview.png    — 전체 경로 조감도"
echo "========================================================"
