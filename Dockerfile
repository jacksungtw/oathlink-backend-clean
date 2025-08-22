# Dockerfile — Railway / Uvicorn
FROM python:3.11-slim

# 可選：供您在部署時注入識別字
ARG BUILD_ID
ENV BUILD_ID=${BUILD_ID}

# 基本環境
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 程式
COPY . .

# Railway 會注入 $PORT；若本機測試就走 8080
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "${PORT:-8080}"]
