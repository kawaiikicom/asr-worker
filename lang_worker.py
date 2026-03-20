"""
Modal Serverless Worker — Language Detection
Whisper tiny (CPU) — определяет язык по первым 30 сек аудио

Deploy:
  modal deploy lang_worker.py
"""

import os
import tempfile
import subprocess

import modal

app = modal.App("lang-worker")

volume = modal.Volume.from_name("asr-models-cache", create_if_missing=True)
CACHE_DIR = "/vol/hf_cache"

# ---------------------------------------------------------------------------
# Image — faster-whisper (CPU)
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install(
        "faster-whisper==1.1.1",
        "huggingface_hub>=0.20.0,<1.0",
    )
)


# ---------------------------------------------------------------------------
# Worker class
# ---------------------------------------------------------------------------

@app.cls(
    image=image,
    volumes={CACHE_DIR: volume},
    scaledown_window=60,
    timeout=120,
    enable_memory_snapshot=True,
)
class LangWorker:

    @modal.enter()
    def load_models(self):
        from faster_whisper import WhisperModel

        os.environ["HF_HOME"] = CACHE_DIR

        print("Loading Whisper tiny...")
        # cpu + int8 — минимальные ресурсы, достаточная точность для языка
        self.model = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",
            download_root=os.path.join(CACHE_DIR, "whisper"),
        )
        print("Whisper tiny loaded. Worker ready.")

    @modal.fastapi_endpoint(method="POST")
    def detect(self, request: dict) -> dict:
        audio_url = request.get("audio_url")

        if not audio_url:
            return {"error": "audio_url is required"}

        tmp_orig = None
        tmp_30s = None
        try:
            # Скачиваем аудио
            suffix = _guess_suffix(audio_url)
            tmp_fd, tmp_orig = tempfile.mkstemp(suffix=suffix)
            os.close(tmp_fd)

            dl = subprocess.run(
                ["ffmpeg", "-y", "-i", audio_url,
                 "-t", "30",          # только первые 30 секунд
                 "-ar", "16000",      # 16kHz — требование Whisper
                 "-ac", "1",          # моно
                 "-f", "wav", tmp_orig],
                capture_output=True,
            )
            if dl.returncode != 0:
                return {"error": f"Download/convert failed: {dl.stderr.decode()[:300]}"}

            file_size = os.path.getsize(tmp_orig)
            if file_size < 1024:
                return {"error": "Audio too short or empty"}

            # Определяем язык
            _, info = self.model.transcribe(
                tmp_orig,
                task="transcribe",
                language=None,        # auto-detect
                beam_size=1,          # быстрее, для языка достаточно
                without_timestamps=True,
            )

            return {
                "language": info.language,
                "probability": round(info.language_probability, 3),
            }

        except Exception as e:
            return {"error": str(e)}
        finally:
            for p in (tmp_orig, tmp_30s):
                if p and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except Exception:
                        pass


def _guess_suffix(url: str) -> str:
    """Вытаскиваем расширение из URL, fallback → .audio"""
    path = url.split("?")[0]
    ext = os.path.splitext(path)[1]
    return ext if ext else ".audio"
