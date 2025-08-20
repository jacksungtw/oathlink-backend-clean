# Dockerfile — 純建置；啟動交由 railway.json
ARG BUILD_ID
ENV BUILD_ID=${BUILD_ID}
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DB_PATH=/data/memory.db

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py storage.py /app/

# SQLite 實體資料夾由 Railway Disk 掛載至 /data
EXPOSE 8080
# 啟動指令改由 railway.json 的 startCommand 控制