FROM python:3.11-slim

WORKDIR /app

# Install ffmpeg for audio conversion
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py ./

EXPOSE 8000

ENV VOLCENGINE_APP_KEY=""
ENV VOLCENGINE_ACCESS_KEY=""
ENV VOLCENGINE_ENDPOINT="wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"
ENV API_KEY=""
ENV MODEL_MAPPING=""

CMD ["python", "server.py"]
