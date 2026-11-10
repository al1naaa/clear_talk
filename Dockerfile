FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_XET=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser \
    && install -d -o appuser -g appuser /data /home/appuser/app /home/appuser/app/models

WORKDIR /home/appuser/app
COPY requirements.txt ./requirements.txt
# CPU wheels keep CUDA libraries out of the inference image.
RUN python -m pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser backend/ ./backend/
COPY --chown=appuser:appuser web/ ./web/
COPY --chown=appuser:appuser deployment/start.py ./deployment/start.py
COPY --chown=appuser:appuser scripts/__init__.py scripts/download_models.py ./scripts/
COPY --chown=appuser:appuser data/demo/ ./data/demo/
# The deny-by-default Docker context admits only the profile and the fine-tuned recognizer;
# the pinned pretrained separator and recognizer are downloaded on first start.
COPY --chown=appuser:appuser models/ ./models/

USER appuser
ENV CLEARTALK_DATA_DIR=/data \
    CLEARTALK_MODELS_DIR=/home/appuser/app/models \
    CLEARTALK_DEPLOYMENT=private \
    CLEARTALK_SECURE_COOKIES=true \
    CLEARTALK_CPU_THREADS=2 \
    CLEARTALK_MAX_PENDING_JOBS=3 \
    CLEARTALK_JOB_TTL_HOURS=24 \
    CLEARTALK_MAX_STORAGE_MB=1024 \
    PORT=7860

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '7860') + '/api/health', timeout=4)" || exit 1
CMD ["python", "deployment/start.py"]
