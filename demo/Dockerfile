# Live-demo API image. Build from the repository root:
#   docker build -f demo/Dockerfile -t traffic-demo .
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./reqs/requirements.txt
COPY demo/requirements.txt ./reqs/demo-requirements.txt
# CPU-only torch keeps the image ~2 GB smaller; the demo server has no GPU.
RUN pip install --no-cache-dir torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r reqs/requirements.txt -r reqs/demo-requirements.txt

COPY solution.py ./
COPY src ./src
COPY weights ./weights
COPY demo ./demo

# CPU-friendly detector settings for the demo (the submission uses 3 / 1280 on a GPU).
ENV JOBS_DIR=/data/jobs PYTHONUNBUFFERED=1 TRAFFIC_DETECT_STRIDE=8 TRAFFIC_DETECT_IMGSZ=800
EXPOSE 8000
CMD ["uvicorn", "demo.app:app", "--host", "0.0.0.0", "--port", "8000"]
