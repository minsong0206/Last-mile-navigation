"""
dashboard_capture.py

배포 스크립트가 시작할 때부터 끝날 때까지, 웹 대시보드(debug_web.py)를 주기적으로
스크린샷해서 하나의 동영상으로 저장한다. docs/deployment_visualization/02_debug_panel/
generate_debug_panel_capture.py와 동일한 방식(pyppeteer 헤드리스 브라우저로 실제
렌더링된 페이지를 그대로 캡처)을 재사용 — 사람이 보는 화면 그대로 남긴다.

2026-10-03: 최초 버전은 GO LIVE 순간 PNG 한 장만 찍었음. 실제 주행 전체 과정을
다시 보고 싶다는 요청으로 "프로세스 시작~종료까지 계속 캡처해서 영상으로" 저장하는
방식으로 변경 — Chrome을 매번 새로 켜면(콜드스타트 수 초, CLAUDE.md 참고) 느려서,
브라우저/페이지를 한 번만 띄워두고 그 안에서 주기적으로 스크린샷만 반복한다.

2026-10-03(2차): Notion에 업로드하면 재생이 안 됨 — 원인은 cv2.VideoWriter의
`mp4v` fourcc가 MPEG-4 Part 2 코덱인데(이 환경의 cv2 ffmpeg 빌드엔 H.264 인코더가
아예 링크 안 돼있어서 `avc1`/`h264` fourcc는 전부 VideoWriter.isOpened()=False로
즉시 실패함, 직접 테스트해서 확인), Notion/대부분의 웹 플레이어는 H.264만 기대함.
그래서 cv2.VideoWriter를 버리고, 시스템에 설치된 `ffmpeg` 바이너리(libx264 포함 —
`ffmpeg -encoders`로 확인됨)를 서브프로세스로 띄워서 raw BGR 프레임을 stdin으로
파이프하고 `-c:v libx264 -pix_fmt yuv420p -movflags +faststart`로 직접 인코딩한다.
yuv420p는 재생 호환성 때문에(libx264 기본값은 입력에 따라 다를 수 있음), faststart는
웹에서 전체 다운로드 전에 미리보기가 되게 하기 위함. **이 머신에 `ffmpeg` 바이너리가
있어야 함** — 없으면 아래 안전 설계대로 조용히 실패하고 로그만 남음(제어 루프엔
영향 없음), 배포 전 `which ffmpeg`로 한 번 확인 권장.

안전 설계: 이건 순수 기록/문서화용이라 실패해도(Chrome 없음, ffmpeg 없음, 타임아웃
등) 제어 루프에는 절대 영향을 주면 안 된다. 그래서 항상 별도 스레드에서
best-effort로 실행하고, 예외는 호출 측 콜백(state.log_error 등)으로만 전달한다 —
절대 raise해서 메인 루프를 죽이지 않음.
"""
import asyncio
import io
import subprocess
import threading
from pathlib import Path

import numpy as np
from PIL import Image

VIEWPORT = {"width": 1500, "height": 1600}  # fullPage 스크린샷 대신 고정 크기 사용
# (영상은 매 프레임이 같은 크기여야 함 — 2026-10-03 실측 페이지 높이 1554px 기준으로
# 여유 있게 1600으로 고정, 넘치면 스크롤 영역은 잘림)
DEFAULT_INTERVAL_S = 2.0  # 캡처 주기(실측정) — 너무 짧으면 Chrome 스크린샷 부하가 커짐


class DashboardRecorder:
    """start()로 백그라운드 스레드에서 녹화 시작, stop()으로 종료+영상 파일 확정.
    프로세스 시작 직후(대시보드 서버가 뜬 직후) start(), run()의 finally에서 stop()
    호출하는 용도 — omnivla_edge_deploy.py 참고."""

    def __init__(self, port, out_dir, label, interval_s=DEFAULT_INTERVAL_S,
                 on_error=None, on_done=None):
        self.port = port
        self.out_path = Path(out_dir) / f"dashboard_{label}.mp4"
        self.interval_s = interval_s
        self.on_error = on_error
        self.on_done = on_done
        self._stop_event = threading.Event()
        self._thread = None
        self._frame_count = 0

    def start(self):
        Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True,
                                         name="dashboard-recorder")
        self._thread.start()
        return self

    def stop(self, join_timeout_s=10.0):
        """논블로킹 호출 측 입장에서 안전하게 쓸 수 있도록 join에 타임아웃을 둠 —
        Chrome 종료가 걸려도 배포 스크립트 전체 종료가 무한정 막히지 않게 하기 위함."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout_s)

    def _run(self):
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._record())
        except Exception as e:
            if self.on_error:
                self.on_error(e)
        finally:
            loop.close()

    async def _record(self):
        from pyppeteer import launch
        # handleSIGINT/SIGTERM/SIGHUP=False 필수 — 백그라운드 스레드에서 실행되는데
        # pyppeteer 기본 동작은 자기 프로세스 종료 시그널을 가로채려 시도하고 그건
        # 메인 스레드에서만 가능해서 "signal only works in main thread" 에러로 실패함(실측).
        browser = await launch(headless=True, args=["--no-sandbox"],
                                executablePath="/usr/bin/google-chrome",
                                handleSIGINT=False, handleSIGTERM=False, handleSIGHUP=False)
        ffmpeg_proc = None
        try:
            page = await browser.newPage()
            await page.setViewport(VIEWPORT)
            await page.goto(f"http://127.0.0.1:{self.port}/", waitUntil="networkidle0", timeout=30000)

            while not self._stop_event.is_set():
                png_bytes = await page.screenshot({"type": "png"})
                frame = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
                frame_bgr = frame[:, :, ::-1]  # RGB -> BGR (ffmpeg rawvideo 입력 포맷에 맞춤)
                if ffmpeg_proc is None:
                    h, w = frame_bgr.shape[:2]
                    fps = 1.0 / self.interval_s
                    ffmpeg_proc = subprocess.Popen(
                        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                         str(self.out_path)],
                        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                ffmpeg_proc.stdin.write(frame_bgr.tobytes())
                self._frame_count += 1
                # time.sleep이 아니라 Event.wait — stop()이 즉시 깨울 수 있게 함
                # (대기 중 stop()이 불리면 interval_s 끝까지 안 기다리고 바로 종료).
                self._stop_event.wait(timeout=self.interval_s)
        finally:
            if ffmpeg_proc is not None:
                ffmpeg_proc.stdin.close()
                ffmpeg_proc.wait(timeout=15)
            await browser.close()
            if self.on_done and self._frame_count > 0:
                self.on_done(self.out_path, self._frame_count)
