# Volcengine ASR OpenAI-Compatible Server

将火山引擎 ASR 封装为 OpenAI Whisper 兼容的 API 服务。

## 快速部署

```bash
git clone <repo-url>
cd volcengine-asr-openai
cp .env.example .env
# 编辑 .env 填入凭证
docker-compose up -d
```

## 调用示例

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="your_api_key")

with open("audio.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=f
    )
print(transcript.text)
```

或使用 curl:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer your_api_key" \
  -F "file=@audio.wav" \
  -F "model=whisper-1"
```

## 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| VOLCENGINE_APP_KEY | ✅ | 火山引擎 App Key |
| VOLCENGINE_ACCESS_KEY | ✅ | 火山引擎 Access Key |
| API_KEY | ❌ | 可选的 API 密钥验证 |
| MODEL_MAPPING | ❌ | 模型映射 JSON |

## 支持的音频格式

支持常见音频格式（wav, mp3, m4a, flac 等），服务会自动转换为所需格式。
