# FrodoBot Mini 배포 전 검증 체크리스트

리뷰 피드백: (1) map-zero 케이스에서 직진해야 하는데 우회전 예측, (2) 데이터셋 직진/좌회전/우회전/커브 비율 확인 필요, (3) 누적 맵 입력/좌표축/스케일 검증, (4) 전체 파이프라인 리뷰, (5) 실배포 테스트.

---

## 우선순위 0 — 즉시 실행 가능 (코드 재작성 불필요)

### 실험 0-1. 데이터셋 시나리오 분포 분석 — **완료**

- **목적**: 파인튜닝에 쓰인 538개 selected segment의 직진/좌회전/우회전/커브 비율을 정량화하여, "map-zero인데 우회전 예측"이 클래스 불균형으로 설명되는지 확인.
- **설계**: `scripts/analysis/analyze_dataset_distribution.py` 실행. GPS 궤적을 2m 간격으로 리샘플링 → heading 변화량 기반으로 8개 시나리오(`straight, curve_left/right, sharp_left/right_turn, winding, intersection, complex_urban`)로 분류. segment 단위와 실제 학습 샘플 가중치인 frame 단위 둘 다 집계.
- **결과** (2026-07-21 재실행, `dataset_analysis/dataset_labels.csv`):

  | 시나리오 | segments | frames | frame % |
  |---|---|---|---|
  | complex_urban | 128 | 147,625 | 32.3% |
  | intersection | 160 | 105,215 | 23.0% |
  | winding | 79 | 103,945 | 22.7% |
  | **straight** | 90 | 48,597 | **10.6%** |
  | sharp_right_turn | 40 | 25,068 | 5.5% |
  | sharp_left_turn | 24 | 15,000 | 3.3% |
  | curve_right | 9 | 6,553 | 1.4% |
  | curve_left | 8 | 5,673 | 1.2% |

  - 스크립트 자체 경고: `Straight-road frames: 10.6% → Dataset is NOT dominated by straight driving.`
  - **좌/우회전 불균형 재확인**: right(`sharp_right_turn`+`curve_right`) : left(`sharp_left_turn`+`curve_left`) = frame 기준 31,621 : 20,673 = **1.53 : 1**. segment 기준도 49:32로 동일 비율.
  - 결론: "직진 프레임이 전체의 10.6%뿐이고, 회전 중에서도 우회전이 1.53배 많다" — map-zero(직진 상황)에서 모델이 소수 클래스인 straight보다 다수 클래스인 turn(그중에서도 right 쪽 우세)을 예측하는 쪽으로 편향될 개연성이 데이터 수준에서 존재함. 이 실험은 **원인을 확정하진 못하지만(상관관계), 가설을 데이터로 뒷받침**함.
  - **이 결과에 대한 이전 세션 후속 코멘트**: "라벨 자체는 괜찮고..." — 이 문장이 중간에 끊겼습니다. 라벨링 로직(`classify_segment()`) 자체의 신뢰도에 대해 추가로 하고 싶은 말씀이 있으면 이어서 알려주세요. (예: 특정 segment의 분류가 이상해 보였다든지, 임계값이 과하다든지)

---

## 실험 0-2. 맵 좌표축/회전 부호 합성 검증 — **완료, 결과: 정상**

- **목적**: `render_frame()`(학습용 맵 생성기, `osm_pipeline/py/osm_map_generator.py:151-236`)의 `rot_deg = 90 - heading_deg` 공식이 실제로 "이미지 위=로봇 전진 방향, 이미지 오른쪽=로봇 우측"이 되도록 맞게 동작하는지 확인. 실제 OSM 타일(네트워크) 없이, 캔버스에 직접 N/S/E/W 나침반 라벨과 좌/우로 굽은 가짜 경로를 그려서 `render_frame()`에 그대로 통과시키는 합성 테스트.
- **설계**: `scripts/analysis/verify_map_coordinate_axes.py` (신규 작성). 두 개의 하위 테스트:
  - **Test 1 (나침반)**: heading을 East/North/West/South 4가지로 바꿔가며, 각 방향에 미리 그려둔 N/S/E/W 라벨이 회전 후 어느 위치(상/하/좌/우)에 오는지 확인.
  - **Test 2 (경로 곡선)**: heading=East로 고정하고, 실제 학습 데이터와 같은 형식의 "미래 경로(빨간선)"를 로봇 기준 좌회전/우회전하는 모양으로 만들어서 `render_frame()`에 통과시킴.
  - 실행: `python3 scripts/analysis/verify_map_coordinate_axes.py`
- **결과**: `attention_analysis/coord_axis_test_compass_combined.png`, `attention_analysis/coord_axis_test_route_combined.png`
  - Test 1: heading=East → 위=E, 왼쪽=N, 오른쪽=S, 아래=W. heading=North/West/South도 전부 "위=현재 진행방향, 왼쪽=로봇 좌측, 오른쪽=로봇 우측" 규칙을 정확히 만족.
  - Test 2: 로봇 기준 우회전하도록 만든 가짜 경로 → 이미지 오른쪽으로 휨. 좌회전 경로 → 이미지 왼쪽으로 휨. **기대한 대로 정상 동작.**
  - **결론: `render_frame()`의 좌표축/회전 공식은 버그가 없습니다.** 이전에 지적했던 `save_bev()`(디버그 전용, `rot_deg = 270 - heading_deg`)와의 180도 차이는 실제로 `save_bev()` 쪽의 표기가 다를 뿐이고(디버그 이미지에만 영향, 모델 입력과 무관), 학습에 실제로 쓰이는 `render_frame()`은 좌표축이 맞습니다.
  - **이 실험이 의미하는 것**: "map-zero인데 우회전 예측" 버그의 원인 후보 중 "좌표축이 뒤집혀서 모델이 배운 좌/우가 실제와 반대"라는 가설은 **기각**됩니다. 남은 유력 후보는 (a) 데이터 클래스 불균형(직진 10.6%, 우회전:좌회전=1.53:1 — 실험 0-1), (b) 배포 시 미래 경로선을 못 그려서 생기는 분포 밖(OOD) 입력 문제 쪽으로 좁혀집니다.

---

## 실험 C 상세 — Straight vs Curve Map Swap Test (`test_map_causality.py --method swap`)

> 사용자 질문: "map이 직진/좌회전 구분 자체가 안 되어서 확인이 잘 안 됨 — 다시 해볼 실험인지, 목적과 결과가 뭔지?"

**목적**: 카메라 관측(obs)은 완전히 고정한 채, **map 입력만** straight ↔ curve로 바꿔치기했을 때 예측 궤적이 실제로 달라지는지 확인. 이게 확인되면 "모델이 map을 단순 노이즈로 무시하는 게 아니라 map의 도로 형태 정보를 실제로 조향에 반영한다"는 causal한 증거가 됩니다. `map-zero 우회전` 버그를 진단하기 전에 먼저 "애초에 모델이 map을 보고 있긴 한가?"를 확인하는 선행 실험입니다.

**현재 설계** (`run_map_swap_test`, `test_map_causality.py:383-502`):
1. `output_rides_00`(파인튜닝에 안 쓰인 별도 라이드)의 segment들을 GPS 곡률로 분류
2. straight 그룹: `curvature_density`(deg/100m, **segment 전체 평균**) 가장 낮은 3개
3. curve 그룹: 같은 지표 가장 높은 3개
4. 각 segment에서 **`frac=0.6` 지점(세그먼트 60% 지점) 딱 한 프레임**의 map PNG만 뽑음 (`pick_map_png`)
5. held-out test set의 obs 1개를 고정하고, 위에서 뽑은 6장의 map을 각각 넣어서 예측 궤적 6개를 비교

**"구분이 안 된다"는 문제의 원인 (코드 확인)**:
- `curvature_density`는 **segment 전체 평균**입니다. 예를 들어 sharp_right_turn으로 분류된 segment도 전체 궤적 중 회전은 짧게 한 번뿐이고 나머지는 직진일 수 있습니다.
- 그런데 대표 프레임은 무조건 `frac=0.6`(세그먼트의 60% 지점) 딱 한 곳만 봅니다 — **그 지점이 실제로 회전 중인 지점이라는 보장이 없습니다.** curve로 분류된 segment라도 60% 지점이 회전 전/후의 직진 구간이면 그 프레임의 map(반경 25m 크롭)은 straight map과 거의 똑같이 보입니다.
- 즉 "segment 라벨(전체 궤적 기준)"과 "그 안에서 뽑은 한 프레임의 로컬 map 내용(반경 25m)"이 불일치할 수 있는 게 설계상 허점입니다. 사용자가 겪은 "직진/좌회전이 구분이 안 됨" 현상은 정확히 이걸로 설명됩니다.

**다시 해볼 가치가 있는가 → 예, 하지만 프레임 선택 로직을 고쳐야 함**:
- `pick_map_png`가 segment 평균이 아니라, **선택된 프레임 자체의 로컬 곡률**(예: 그 지점 앞뒤 ±25m 구간의 heading 변화량)을 기준으로 "회전이 실제로 지금 이 map 크롭 안에서 보이는 지점"을 고르도록 수정 필요.
- 대안: `frac`을 여러 개(0.3, 0.5, 0.7 등) 시도해서 그중 map 크롭 안에 가장 뚜렷한 곡선이 보이는 프레임을 자동 선택.
- 이미 `run_matched_sequence_test`(실험 E, `_run_one_sequence`)가 한 segment 안에서 여러 시점(frac 0.15~0.85)을 연속으로 뽑아 보여주므로, 이 결과(`attention_analysis/rides00_sequence_straight.png`, `rides00_sequence_curve.png`)를 먼저 보면 "어느 frac 지점이 실제로 휘어 보이는지" 육안으로 고를 수 있습니다 — swap 테스트를 고치기 전에 이 시퀀스 결과부터 확인하는 걸 권장.
- **기대 결과(정상이라면)**: map이 실제로 곡선을 보여주는 프레임끼리 비교했을 때, straight map → 예측 궤적의 누적 lateral offset ≈ 0, curve map → 0이 아닌 방향성 있는 offset. 스크립트가 이미 이 수치(`straight map → 평균 누적 lateral offset`, `curve map → ...`)를 출력하도록 되어 있으니, 프레임 선택만 고치면 바로 정량적 판단 가능.

---

## 추가로 확인이 필요한 부분 (이전 답변에서 애매했던 점)

이전 메시지에서 4번째 항목("해당 실험은 내가 제대로 이해하지 못한 거 같아 다시 봐야 할 듯")이 어떤 실험을 가리키는지 문장이 끊겨서 확실치 않습니다. 후보는:
- 위에서 다룬 실험 C(map swap test) — 같은 얘기의 반복이라면 위 설명으로 답이 됐을 것
- 실험 A(`run_heldout_ablation`, map=zeros/noise/shuffled ablation) — map을 아예 지웠을 때 ADE가 나빠지는지 보는 실험
- 실험 D/E(matched obs+map test/sequence) — 실제 카메라+map을 함께 넣었을 때의 generalization 확인

어느 실험을 다시 보고 싶으신지 알려주시면 그 실험만 짚어서 목적/설계/결과를 다시 설명해드리겠습니다.

---

## Priority 1~5 (요약, 상세는 이전 대화 참고)

1. **데이터셋/맵 입력 검증**: 좌우 flip augmentation 부재, `save_bev()`(270-heading)와 `render_frame()`(90-heading) 180도 회전 불일치, 맵-카메라 인덱스 정합성
2. **파인튜닝/체크포인트 검증**: 9ch(`rides11_finetune.yaml`) vs 3ch odom(`rides11_finetune_odom.yaml`) 중 실제 사용 체크포인트 확정, 프레임 단위 random_split(seed=42)으로 인한 val/test leakage 가능성
3. **오프라인 추론/ablation**: `test_map_causality.py` A(heldout)/C(swap, 위 상세)/D/E, `check_map_ablation.py`, `check_map_attention.py`
4. **서버-로봇 통신**: OmniVLA-Edge용 실시간 연동 코드 부재 — `deployment/LogoNav_frodobot.py` 구조를 참고해 새로 작성 필요
5. **실배포**: 학습 시 map에 그려지는 "GT 미래 경로(빨간선)"를 실로봇에서 어떻게 대체할지가 핵심 미해결 과제 — 대체 없이 배포하면 OOD 입력이 되어 이번 우회전 버그와 동일한 실패 모드 재현 가능성 높음
