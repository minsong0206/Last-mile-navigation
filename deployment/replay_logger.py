"""
replay_logger.py

Deterministic replay logging for the OmniVLA-Edge 실배포 파이프라인.

배경: 2026-09-18 세션에서 "이상 행동이 관찰된 순간 모델이 실제로 뭘 봤는지" 재현을
시도했다가, 대시보드 카메라(`debug_web.py`의 `/frame.jpg`, 독립적인 ~1Hz 갱신)로
당시 프레임을 근사 복원했더니 그날 실제로 기록된 모델 출력(`pred_xy_m`)과 전혀
안 맞았다 (target y: 재구성 -0.05 vs 실제 로그 +1.14). 대시보드 스트림과 추론에
실제로 쓰인 `frame_buffer`는 서로 독립적으로 샘플링되는 별개의 스트림이라, "화면에
보이던 것"이 "그 순간 모델에 들어간 바이트"라는 보장이 없었던 것.

그래서 여기서는 대시보드가 아니라 `OmniVLAEdgeDeployment.build_inputs()`가
`obs_stack`/`map_tensor`를 만드는 데 실제로 사용하는 PIL 이미지 자체를 디스크에
저장한다. 다음에 이상 행동이 다시 발생하면, 저장된 이 파일들 + 저장된 heading/
route/tick_id로 100% 동일한 offline forward pass를 재현할 수 있다.

저장 정책: "항상 저장"을 기본으로 한다 — 컨텍스트 프레임은 새 프레임일 때만
저장(중복 저장 안 함, maybe_update_frame_buffer()의 is_new 플래그 재사용)하고,
지도는 ego 위치/heading이 매 틱 달라지므로 항상 저장한다. 컨텍스트 프레임은
obs_transform(96x96 리사이즈) "이전"의 원본 해상도 그대로 JPEG q90으로 저장한다
— 리사이즈는 model_omnivla_edge_odom 입력을 만들 때 결정론적으로 재현 가능해서
저장 단계에서 미리 줄일 이유가 없고, 사람이 나중에 이 이미지를 직접 봤을 때
(예: "그때 장애물이 있었나?") 96x96은 너무 작아서 알아보기 어렵기 때문.

용량: 2026-09-25 오프라인 스모크 테스트(offline_smoke_test.py, 화면녹화에서 뽑은
~640x362 카메라 크롭 사용)에서 15틱 처리에 컨텍스트 프레임 13개+지도 8개 = 21개
파일, 총 788KB(평균 파일당 ~37.5KB) 실측. 단, 이건 실제 SDK `/v2/front` 원본
해상도로는 검증 못 했다(오늘 세션엔 SDK 연결이 없었음 — 이전 세션에서 실측한
base64 길이 기준 원본 프레임이 이보다 훨씬 클 수 있음, 대략 100~400KB/프레임
추정). 그러면 10~15분 테스트(카메라 갱신 ~1.8Hz, 루프 ~1~2Hz 기준)는 대략
150~350MB/run 정도로 추정 — SSD 쓰기 부담은 여전히 무시 가능(수 ms)하지만,
누적 용량은 위 추정보다 클 수 있으니 **내일 첫 --dry_run 후 실제 용량을 확인하고
필요하면 오래된 run 폴더를 정리**할 것. 이벤트 기반(|angular|>=임계값 등만 저장)
방식은 채택하지 않음 — "평범해 보였는데 나중에 문제였던" 구간을 사전에 걸러내다
놓칠 위험이 더 크다고 판단함(정확히 5번 replay 실패의 원인이 "그 순간 데이터가
없어서"였음). 대신
event_flag는 메타데이터 필드로만 남겨서 나중에 필터링을 쉽게 한다.
"""
from pathlib import Path

import numpy as np
from PIL import Image


class ReplayLogger:
    def __init__(self, run_id: str, base_log_dir: Path):
        self.run_id = run_id
        self.frames_dir = Path(base_log_dir) / "frames" / run_id
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._next_frame_id = 0
        self._last_frame_id = None  # 가장 최근에 저장(또는 재사용)한 컨텍스트 frame_id
        self._next_tick_id = 0

    def next_tick_id(self) -> int:
        """step() 시작 시 1번 호출 — JSONL 레코드와 저장 파일을 잇는 공통 키."""
        tick_id = self._next_tick_id
        self._next_tick_id += 1
        return tick_id

    def record_context_frame(self, pil_img: Image.Image, is_new: bool) -> int:
        """새 카메라 프레임이면(maybe_update_frame_buffer()가 판단) 디스크에 저장하고
        새 frame_id를 발급, 아니면(직전과 같은 프레임을 버퍼에 다시 넣은 경우) 마지막
        frame_id를 그대로 재사용한다 — 카메라 실측 갱신률(~1.8Hz)이 폴링 주기(3Hz
        목표)보다 느려서 같은 프레임이 여러 틱에 걸쳐 재사용되는 경우가 흔하므로,
        매번 저장하면 디스크만 낭비하고 "몇 번이나 같은 프레임이 재사용됐는지"라는
        중요한 진단 정보도 사라진다 — 그래서 dedup 자체를 replay 로그의 핵심 설계로
        삼는다."""
        if is_new or self._last_frame_id is None:
            frame_id = self._next_frame_id
            self._next_frame_id += 1
            path = self.frames_dir / f"ctx_{frame_id:06d}.jpg"
            pil_img.convert("RGB").save(path, format="JPEG", quality=90)
            self._last_frame_id = frame_id
        return self._last_frame_id

    def save_map(self, map_np: np.ndarray, tick_id: int) -> str:
        """모델 입력 지도(transform 적용 전, render_frame() 직후의 RGB 배열)를 저장.
        ego 위치/heading이 매 틱 다르므로 항상 저장(dedup 안 함). run_id 기준
        상대경로를 반환해서 JSONL에 그대로 남기기 쉽게 한다."""
        rel_path = f"map_{tick_id:06d}.png"
        Image.fromarray(map_np).save(self.frames_dir / rel_path)
        return f"frames/{self.run_id}/{rel_path}"

    def context_frame_path(self, frame_id: int) -> str:
        return f"frames/{self.run_id}/ctx_{frame_id:06d}.jpg"
