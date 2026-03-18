# RunPod base image — PyTorch 2.2.0 + CUDA 12.1 + Python 3.10
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Fix torchvision/transformers version conflicts
RUN pip install --no-cache-dir \
    torchvision==0.17.0 \
    transformers==4.39.3 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download all models from HuggingFace into the image
# HF_TOKEN needed for pyannote (accept terms at hf.co first:
#   pyannote/speaker-diarization-3.1
#   pyannote/segmentation-3.0)
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

RUN python -c "\
import whisperx; \
whisperx.load_model('tiny', 'cpu', compute_type='float32'); \
whisperx.load_model('large-v3', 'cpu', compute_type='float32'); \
"

RUN python -c "\
import gigaam; \
gigaam.load_model('v3_e2e_rnnt'); \
"

RUN if [ -n "$HF_TOKEN" ]; then python -c "\
from pyannote.audio import Pipeline; \
Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token='$HF_TOKEN'); \
"; fi

# Copy worker code
COPY handler.py .

CMD ["python", "-u", "handler.py"]
