# RunPod 파인튜닝 셋업 진행 상황 (2026-09-06)

## 2026-09-07 밤 ~ 09-08: 자율 실행 재시도 (사용자 수면 중, Claude가 전체 파이프라인 재실행)

- 1차 재시도(volume `8z9ehr268k`, pod `vy4oxjt6uz0j7j`)에서 데이터 업로드/압축해제까지는
  성공했으나, **학습 시작 직후 크래시** — 일부 프레임의 JPEG가 없고 mp4 폴백도 없어서
  (`videos/` 원본 비디오를 안 올렸음) `FileNotFoundError`로 전체 학습이 죽음.
  자동 정리 로직은 정상 작동해서 pod은 즉시 삭제됨(과금 안전).
- **원인 수정**: `osm_pipeline/py/rides11_dataset.py::_get_frame()`이 JPEG도 mp4도 없을 때
  같은 episode의 가장 가까운 JPEG로 대체하도록 수정 (커밋 `ee3419e`). 크래시 대신
  드문 결손 프레임을 근사치로 넘어가게 함.
- **2차 시도**: 코드 수정 반영 후 volume(`3ajx1a913l`)/pod(`frodobot-finetune-3`) 새로 만들어서
  처음부터 재실행 중 — 진행상황은 `C:\Users\minso\AppData\Local\Temp\claude\full_run.log`에
  누적됨 (이 파일은 로컬 임시폴더라 재부팅하면 사라질 수 있음, 중요 내용은 이 문서에 옮겨적을 것).
- 이번엔 pod 생성 실패("no instances"/"ssh 영영 안 뜨는 불량 머신") 케이스도 스크립트가
  자동으로 감지해서 삭제 후 재시도하도록 만들어둠 (최대 6회 시도).
- **다음 세션이 확인할 것**: `full_run.log`에서 `=== FULL RUN END (status=...) ===` 줄을 찾을 것.
  `status=done`이면 성공 (`D:\best_omnivla_edge_rides11_odom_20260906.pth`에 체크포인트 저장됨).
  `status=error`/`stalled_or_exited`/`crashed_immediately`면 로그에서 원인 파악 후 재시도 필요.
  pod은 끝나면(성공/실패 상관없이) 스크립트가 알아서 삭제하므로, `runpodctl pod list`가
  비어있다면 실행이 끝났다는 뜻(로그로 성공/실패만 판단하면 됨).

## ⚠ 2026-09-07: 전부 삭제함 — 이 문서는 "재현 절차"로만 참고할 것

비용 문제로 **pod(`vbfkj917vm5mkr`)와 network volume(`0yymo3grjw`) 둘 다 삭제 완료**
(`runpodctl pod list` / `network-volume list` 둘 다 빈 목록으로 확인됨). 아래 내용에 나오는
pod id/volume id/IP/포트는 전부 **더 이상 유효하지 않음** — 다시 시작하려면 "다음에 할 일"이
아니라 **맨 처음부터**(volume 생성 → S3 업로드 → pod 생성 → 압축 해제) 다시 해야 함.

- 삭제 이유: pod을 34시간 켜둔 채로 방치해서 GPU 과금(~$25)이 쌓였고, network volume도
  pod을 꺼도 별도로 계속 과금(100GB × $0.07/GB/월 ≈ $7/월)된다는 걸 확인 → 사용자가 전부
  정리하기로 결정함.
- **재개 시 필요한 것**: 로컬(`D:\osm_maps_arrow_20260906`, `E:\CAPSTONE\frames_output_rides_11.tar`
  등)에 원본 데이터가 남아있으면 그대로 재사용 가능 — 그렇다면 "지금까지 한 것" 절차를
  그대로 다시 따라가면 됨 (volume 생성부터). 로컬 원본도 지웠다면 osm 맵 재생성부터 다시.
- **배운 것(아래 "이번 세션에 배운 것" 절)은 여전히 유효**하니 재시도할 때 그대로 활용할 것
  (특히 pod은 다 쓰면 바로 stop/remove, volume도 필요 없어지면 바로 delete — 켜둔 채
  방치하지 않기).

---

**(아래는 삭제 전 시점의 기록 — 재현 시 참고용)**

**세션이 끊겨도(와이파이/배터리) 여기서 이어갈 것.** 아래 순서대로 확인하면 됨.

## 지금까지 한 것 (전부 완료됨)

1. **RunPod 인증**: `runpodctl`(Windows용, `C:\Users\minso\AppData\Local\Programs\runpodctl\runpodctl.exe`)
   설치 + API 키 `~/.runpod/config.toml`에 저장 완료 (계정: ksh030630@naver.com, 잔액 ~$100).
   SSH 키도 자동 생성/등록됨(`C:\Users\minso\.runpod\ssh\runpodctl-ssh-key`).
2. **AWS CLI**: 관리자 권한 없어서 MSI 설치 실패 → **pip으로 설치함**
   (`python -m awscli`, `C:\Users\minso\AppData\Local\Programs\Python\Python313\python.exe -m awscli`).
   S3 자격증명은 `~/.aws/credentials`의 `[runpod]` 프로필에 저장됨 (access key
   `user_3FtO3NTYD60FEGgAriasHrVqgPZ` + 별도 secret, 값 자체는 이 파일에 안 적어둠 —
   필요하면 RunPod 콘솔 → Settings → S3 API Keys에서 재발급). `~/.aws/config`의
   `[profile runpod]`에 `multipart_chunksize = 64MB` 설정해둠 (아래 "배운 것" 참고).
3. **Network Volume 생성**: id **`0yymo3grjw`**, 이름 `frodobot-training`, **100GB**, **EU-RO-1**.
   (볼륨은 이 데이터센터에 고정 — pod도 반드시 EU-RO-1에 만들어야 함)
4. **데이터 업로드 완료** (전부 `s3://0yymo3grjw/`에 있고, pod에서는 `/workspace/`에 그대로 보임):
   - `checkpoints/omnivla-edge-odom3ch.pth` (414MB) — 파인튜닝 시작 체크포인트
   - `data/train/data-00000-of-00001.arrow` (374MB) — GPS/heading/waypoint 원본
   - `data/frames_output_rides_11.tar` (56.1GB) — 카메라 이미지 원본 (압축 안 푼 tar)
   - `data/osm_maps_arrow_20260906.tar` (7.1GB) — **이번 세션에 새로 만든 OSM 맵**
     (`docs/0905.md`/`0906.md`에서 확정한 B안 스케일 + cartocdn 타일)
5. **GPU Pod 생성 완료**: id **`vbfkj917vm5mkr`**, 이름 `frodobot-finetune`,
   **RTX 4090, EU-RO-1**, `/workspace`에 위 network volume 마운트됨,
   `runpod-torch-v280` 템플릿 (PyTorch 2.8.0, CUDA 12.8, Python 3.12).
   - **SSH 접속**:
     ```bash
     ssh -i "C:\Users\minso\.runpod\ssh\runpodctl-ssh-key" -o StrictHostKeyChecking=no -p 30761 root@213.173.111.15
     ```
     (IP/포트는 pod 재시작하면 바뀔 수 있음 — 바뀌면
     `MSYS_NO_PATHCONV=1 runpodctl ssh info vbfkj917vm5mkr` 로 새로 확인)
   - 비용: $0.74/hr (secure cloud)
6. **레포 clone 완료**: `/workspace/Last-mile-navigation` (GitHub main 브랜치, 이번 세션에
   push한 최신 커밋 `efd13e0` 포함 — 맵 스케일/타일/heading 수정 전부 반영된 버전).
7. **파이썬 패키지 설치 완료** (템플릿의 system python에 `--break-system-packages`로):
   `efficientnet_pytorch openai-clip ftfy regex wandb pyarrow av opencv-python-headless
   matplotlib pyyaml einops` (torch/torchvision은 템플릿에 이미 있음). 전부 import 확인함.
8. **tar 압축 해제 진행 중** (백그라운드, `setsid`로 detach돼서 SSH 끊겨도 안 죽음):
   - `/workspace/extract_osm.log` — `osm_maps_arrow_20260906.tar` 푸는 중
     (완료되면 파일 끝에 `EXTRACT_DONE` 찍힘)
   - `/workspace/extract_frames.log` — `frames_output_rides_11.tar` 푸는 중
     (완료되면 파일 끝에 `EXTRACT_DONE` 찍힘)
   - **진행상황 확인 명령** (SSH로):
     ```bash
     ssh -i "C:\Users\minso\.runpod\ssh\runpodctl-ssh-key" -o StrictHostKeyChecking=no -p 30761 root@213.173.111.15 \
       'grep -c EXTRACT_DONE /workspace/extract_osm.log /workspace/extract_frames.log; \
        find /workspace/osm_maps_arrow_20260906 -mindepth 1 -maxdepth 1 -type d | wc -l; \
        find /workspace/data/frames -mindepth 1 -maxdepth 1 -type d | wc -l'
     ```
     (목표: osm 538개 세그먼트 폴더, frames 331개 episode 폴더)
   - tar 안의 uid/gid 관련 경고("Cannot change ownership...")는 **무시해도 됨** — Windows에서
     만든 tar라 Linux uid로 못 바꾸는 것뿐, 파일 내용 자체는 정상 추출됨.

## 다음에 할 일 (순서대로)

1. **압축 해제 완료 확인** (위 명령으로 두 log 다 `EXTRACT_DONE` 나올 때까지).
2. **tar 파일 삭제해서 공간 확보** (중요! 100GB 볼륨에 압축본+원본 둘 다 있으면 공간
   부족해질 수 있음 — 압축 해제 확인되면 바로):
   ```bash
   rm /workspace/data/frames_output_rides_11.tar /workspace/data/osm_maps_arrow_20260906.tar
   ```
3. **RunPod용 config yaml 새로 작성** (`config/rides11_finetune_odom_20m.yaml` 참고해서
   경로만 교체 — 원본은 전부 `/media/ms/WD_BLACK_4TB/...` 절대경로라 그대로 못 씀):
   ```yaml
   ckpt_path:   "/workspace/checkpoints/omnivla-edge-odom3ch.pth"
   arrow_path:  "/workspace/data/train/data-00000-of-00001.arrow"
   scores_path: "/workspace/Last-mile-navigation/osm_pipeline/osm_data/output_rides_11/episode_scores.json"
   osm_root:    "/workspace/osm_maps_arrow_20260906"
   video_root:  "/workspace/data"   # frames/episode_XXXX/*.jpg가 이 밑에 옴
   save_dir:    "/workspace/checkpoints/omnivla_edge_rides11_odom_20260906"
   map_range_m: 20.0
   gpu: 0        # ⚠ 원본 코드 기본값은 gpu:1(원저자 2-GPU 머신 기준) — RunPod pod는
                 #   GPU 1장뿐이라 cuda:0이어야 함. 안 바꿔도 fallback으로 cuda:0을
                 #   쓰긴 하지만 경고 뜸, 명시적으로 0으로 설정할 것.
   freeze: "partial"
   epochs: 20
   batch_size: 32
   lr: 1.0e-4
   weight_decay: 1.0e-4
   val_ratio: 0.1
   test_ratio: 0.1
   num_workers: 4
   smooth_weight: 0.1
   save_freq: 5
   log_freq: 50
   vis_freq: 1
   use_wandb: true   # 아래 4번 참고 — WANDB_API_KEY 필요
   wandb_project: "omnivla-edge-rides11-odom"
   run_name: "odom3ch_bs32_partial_map20260906_runpod"
   ```
4. **wandb 로그인 필요 여부 확인** — `use_wandb: true`로 하려면 pod에서
   `export WANDB_API_KEY=<키>` 필요 (사용자에게 물어볼 것, https://wandb.ai/authorize 에서
   확인 가능). 귀찮으면 `use_wandb: false`로 끄고 진행해도 학습 자체는 됨 (로그/시각화만 없음).
5. **학습 시작** (detach 필수, SSH 끊겨도 계속 돌게):
   ```bash
   ssh -i "C:\Users\minso\.runpod\ssh\runpodctl-ssh-key" -o StrictHostKeyChecking=no -p 30761 root@213.173.111.15 \
     'cd /workspace/Last-mile-navigation && setsid bash -c \
      "python3 scripts/omnivla/finetune_omnivla_edge.py --config config/<새로만든yaml>" \
      > /workspace/train.log 2>&1 </dev/null & echo LAUNCHED'
   ```
6. **모니터링** (별도 SSH 호출로, 예상 20 epoch × ~23분/epoch(RTX4090 기준) ≈ 7.7시간):
   ```bash
   ssh ... 'tail -n 30 /workspace/train.log'
   ssh ... 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'
   ```
7. **완료 후**: `best.pth`가 `/workspace/checkpoints/omnivla_edge_rides11_odom_20260906/best.pth`에
   생김 — network volume 위라 pod을 지워도 안 없어짐. 확인되면:
   ```bash
   runpodctl pod remove vbfkj917vm5mkr   # GPU 과금 중지 (volume은 유지됨)
   ```

## 이번 세션에 배운 것 (재발 방지)

- **Windows git-bash에서 `runpodctl`에 `/workspace` 같은 유닉스 경로를 넘기면 자동으로
  `C:/Program Files/...` 식으로 잘못 변환됨** — `MSYS_NO_PATHCONV=1` 접두어를 꼭 붙일 것
  (실제로 한 번 이렇게 잘못 만들어서 pod 지우고 다시 만들었음).
- **RunPod S3 호환 API의 멀티파트 업로드 파트 크기**: 기본값(파트 많음, ~8MB×~7000개)은
  마지막 CompleteMultipartUpload에서 "파트 없음" 에러로 실패, 200MB는 413(Too Large)로
  실패 → **64MB가 안전하게 작동함** (`~/.aws/config`에 이미 설정해둠).
- **RunPod S3 호환 스토리지는 멀티파트 업로드 조각을 자동 정리 안 함** —
  `.s3compat_uploads/` 밑에 그대로 남아서 용량을 이중으로 잡아먹음. 큰 파일 업로드
  후엔 `aws s3 rm --recursive s3://<volume-id>/.s3compat_uploads/`로 수동 정리 필요
  (이미 한 번 정리함, 앞으로 추가 업로드하면 또 확인할 것).
- **pod 생성 시 SSH가 영영 안 뜨는 "불량 머신" 뽑힐 수 있음** (`ssh port not allocated yet`가
  5분 넘게 지속) — 기다리지 말고 `pod delete` 후 재생성. 이번 세션에 2번 겪음, 3번째에 성공.
- **45만 개짜리 작은 파일을 S3에 개별 업로드/동기화하는 건 파일당 오버헤드 때문에
  극단적으로 느림** (13시간+ 예상) — 반드시 로컬에서 tar로 묶은 다음 파일 하나로
  업로드할 것. 반대로 tar 자체를 로컬에서 만드는 것도(45만 개 파일 압축) 디스크 I/O
  때문에 꽤 오래 걸림(1시간 안팎) — 조급해하지 말 것.
- **Seoul OSRM 데이터는 학습(RunPod)엔 전혀 필요 없음** — 실로봇 배포 시
  `deployment/build_live_map.py`가 실시간 경로계산에만 씀. RunPod 업로드 목록에서 제외함.
- 백그라운드 작업 완료 알림이 가끔 실제보다 먼저/늦게 오는 경우가 있었음 (특히 bash
  안에서 또 `&`로 이중 백그라운드 처리했을 때) — **의심되면 항상 S3 listing이나 실제
  파일 크기로 직접 재확인할 것**, 로그/알림만 믿지 말 것.

## 참고 — 관련 문서

- `docs/0905.md`: 맵 스케일(B안)/waypoint horizon(5m)/heading 소스 전환 — 코드 변경 배경.
- `docs/0906.md`: cartocdn 타일 API 키 문제 발견/해결 경위.
- 이 문서(`0906_runpod_setup.md`): 그 결과물(새 OSM 맵)을 RunPod에 올려서 파인튜닝
  준비하는 인프라 작업 기록.
