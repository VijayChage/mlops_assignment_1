FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY data ./data
COPY artifacts ./artifacts
COPY reports ./reports
# Train inside the image for a self-contained coursework demo. Requires network during build.
RUN python scripts/download_data.py && python scripts/train.py
EXPOSE 8000
CMD ["uvicorn", "src.heartml.api:app", "--host", "0.0.0.0", "--port", "8000"]
