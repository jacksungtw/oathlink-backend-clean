FROM python:3.11-slim

ARG BUILD_ID
ENV BUILD_ID=${BUILD_ID}

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# 這裡改成用 shell，確保 $PORT 被替換成數字
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]