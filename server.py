#!/usr/bin/env python3
"""
OpenAI-compatible ASR (Whisper) server wrapping Volcengine ASR API.

Usage:
    python server.py --app-key YOUR_APP_KEY --access-key YOUR_ACCESS_KEY

Environment variables (for Docker):
    VOLCENGINE_APP_KEY, VOLCENGINE_ACCESS_KEY, API_KEY, MODEL_MAPPING

Then call:
    POST http://localhost:8000/v1/audio/transcriptions
    - file: audio file (multipart/form-data)
    - model: whisper-1 (optional)
    - language: zh (optional)
"""
import asyncio
import gzip
import json
import logging
import os
import struct
import subprocess
import tempfile
import uuid
from typing import Any, Dict, List, Optional

import aiohttp
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Volcengine ASR OpenAI-Compatible Server")

# Global config
CONFIG = {
    "app_key": "",
    "access_key": "",
    "endpoint": "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel",
    "api_key": "",
    "model_mapping": {},
}

DEFAULT_MODEL_MAPPING = {
    "whisper-1": "bigmodel",
    "whisper-large": "bigmodel",
}

DEFAULT_SAMPLE_RATE = 16000


# Protocol constants
class ProtocolVersion:
    V1 = 0b0001

class MessageType:
    CLIENT_FULL_REQUEST = 0b0001
    CLIENT_AUDIO_ONLY_REQUEST = 0b0010
    SERVER_FULL_RESPONSE = 0b1001
    SERVER_ERROR_RESPONSE = 0b1111

class MessageTypeSpecificFlags:
    NO_SEQUENCE = 0b0000
    POS_SEQUENCE = 0b0001
    NEG_SEQUENCE = 0b0010
    NEG_WITH_SEQUENCE = 0b0011

class SerializationType:
    NO_SERIALIZATION = 0b0000
    JSON = 0b0001

class CompressionType:
    GZIP = 0b0001


def gzip_compress(data: bytes) -> bytes:
    return gzip.compress(data)

def gzip_decompress(data: bytes) -> bytes:
    return gzip.decompress(data)


def convert_audio_to_wav(audio_bytes: bytes, suffix: str = ".wav") -> bytes:
    """Convert audio to WAV format using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name

    try:
        cmd = [
            "ffmpeg", "-v", "quiet", "-y", "-i", tmp_in_path,
            "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(DEFAULT_SAMPLE_RATE),
            "-f", "wav", "-"
        ]
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Audio conversion failed: {e.stderr.decode()}")
    finally:
        os.unlink(tmp_in_path)


def read_wav_info(data: bytes) -> tuple:
    """Parse WAV header and return audio info."""
    if len(data) < 44 or data[:4] != b'RIFF' or data[8:12] != b'WAVE':
        raise ValueError("Invalid WAV file")

    num_channels = struct.unpack('<H', data[22:24])[0]
    sample_rate = struct.unpack('<I', data[24:28])[0]
    bits_per_sample = struct.unpack('<H', data[34:36])[0]

    # Find data chunk
    pos = 36
    while pos < len(data) - 8:
        subchunk_id = data[pos:pos+4]
        subchunk_size = struct.unpack('<I', data[pos+4:pos+8])[0]
        if subchunk_id == b'data':
            wave_data = data[pos+8:pos+8+subchunk_size]
            return num_channels, bits_per_sample // 8, sample_rate, wave_data
        pos += 8 + subchunk_size

    raise ValueError("Invalid WAV file: no data chunk")


def is_wav(data: bytes) -> bool:
    return len(data) >= 44 and data[:4] == b'RIFF' and data[8:12] == b'WAVE'


def build_auth_headers() -> Dict[str, str]:
    """Build authentication headers for WebSocket connection."""
    return {
        "X-Api-Resource-Id": "volc.bigasr.sauc.duration",
        "X-Api-Request-Id": str(uuid.uuid4()),
        "X-Api-Access-Key": CONFIG["access_key"],
        "X-Api-App-Key": CONFIG["app_key"],
    }


def build_full_client_request(seq: int, language: str = "zh") -> bytes:
    """Build initial full client request."""
    header = bytearray()
    header.append((ProtocolVersion.V1 << 4) | 1)
    header.append((MessageType.CLIENT_FULL_REQUEST << 4) | MessageTypeSpecificFlags.POS_SEQUENCE)
    header.append((SerializationType.JSON << 4) | CompressionType.GZIP)
    header.append(0x00)

    payload = {
        "user": {"uid": "openai_compat_user"},
        "audio": {
            "format": "wav",
            "codec": "raw",
            "rate": DEFAULT_SAMPLE_RATE,
            "bits": 16,
            "channel": 1,
        },
        "request": {
            "model_name": "bigmodel",
            "enable_itn": True,
            "enable_punc": True,
            "enable_ddc": True,
            "show_utterances": True,
        },
    }

    payload_bytes = json.dumps(payload).encode("utf-8")
    compressed = gzip_compress(payload_bytes)

    request = bytearray()
    request.extend(header)
    request.extend(struct.pack(">i", seq))
    request.extend(struct.pack(">I", len(compressed)))
    request.extend(compressed)
    return bytes(request)


def build_audio_request(seq: int, segment: bytes, is_last: bool = False) -> bytes:
    """Build audio-only request."""
    header = bytearray()
    header.append((ProtocolVersion.V1 << 4) | 1)

    if is_last:
        flags = MessageTypeSpecificFlags.NEG_WITH_SEQUENCE
        seq = -seq
    else:
        flags = MessageTypeSpecificFlags.POS_SEQUENCE

    header.append((MessageType.CLIENT_AUDIO_ONLY_REQUEST << 4) | flags)
    header.append((SerializationType.JSON << 4) | CompressionType.GZIP)
    header.append(0x00)

    compressed = gzip_compress(segment)

    request = bytearray()
    request.extend(header)
    request.extend(struct.pack(">i", seq))
    request.extend(struct.pack(">I", len(compressed)))
    request.extend(compressed)
    return bytes(request)


def parse_response(msg: bytes) -> Dict[str, Any]:
    """Parse ASR response."""
    result = {"code": 0, "is_last": False, "text": "", "payload": None}

    header_size = msg[0] & 0x0F
    message_type = msg[1] >> 4
    flags = msg[1] & 0x0F
    compression = msg[2] & 0x0F

    payload = msg[header_size * 4:]

    if flags & 0x01:  # Has sequence
        payload = payload[4:]
    if flags & 0x02:  # Is last
        result["is_last"] = True
    if flags & 0x04:  # Has event
        payload = payload[4:]

    if message_type == MessageType.SERVER_FULL_RESPONSE:
        payload_size = struct.unpack(">I", payload[:4])[0]
        payload = payload[4:]
    elif message_type == MessageType.SERVER_ERROR_RESPONSE:
        result["code"] = struct.unpack(">i", payload[:4])[0]
        payload_size = struct.unpack(">I", payload[4:8])[0]
        payload = payload[8:]

    if payload and compression == CompressionType.GZIP:
        try:
            payload = gzip_decompress(payload)
            result["payload"] = json.loads(payload.decode("utf-8"))
        except Exception:
            pass

    return result


async def transcribe_audio(audio_data: bytes, language: str = "zh") -> str:
    """Transcribe audio using Volcengine ASR."""
    # Convert to WAV if needed
    if not is_wav(audio_data):
        audio_data = convert_audio_to_wav(audio_data)

    # Parse WAV and get audio samples
    num_channels, samp_width, sample_rate, wave_data = read_wav_info(audio_data)

    # Calculate segment size (200ms)
    segment_duration_ms = 200
    size_per_sec = num_channels * samp_width * sample_rate
    segment_size = size_per_sec * segment_duration_ms // 1000

    # Split audio into segments
    segments = []
    for i in range(0, len(wave_data), segment_size):
        segments.append(wave_data[i:i + segment_size])

    headers = build_auth_headers()
    final_text = ""

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(CONFIG["endpoint"], headers=headers) as ws:
            # Send initial request
            seq = 1
            await ws.send_bytes(build_full_client_request(seq, language))
            seq += 1

            # Wait for initial response
            msg = await ws.receive()
            if msg.type != aiohttp.WSMsgType.BINARY:
                raise RuntimeError("Unexpected response type")

            # Send audio segments
            for i, segment in enumerate(segments):
                is_last = i == len(segments) - 1
                await ws.send_bytes(build_audio_request(seq, segment, is_last))
                if not is_last:
                    seq += 1
                await asyncio.sleep(segment_duration_ms / 1000)

            # Receive all responses
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    result = parse_response(msg.data)
                    if result["payload"]:
                        payload = result["payload"]
                        # Extract text from response
                        if "result" in payload:
                            res = payload["result"]
                            if isinstance(res, list) and res:
                                final_text = res[-1].get("text", "")
                            elif isinstance(res, dict):
                                final_text = res.get("text", "")
                        if "text" in payload:
                            final_text = payload["text"]

                    if result["is_last"] or result["code"] != 0:
                        break
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    break

    return final_text


def load_config_from_env():
    """Load configuration from environment variables."""
    CONFIG["app_key"] = os.getenv("VOLCENGINE_APP_KEY", "")
    CONFIG["access_key"] = os.getenv("VOLCENGINE_ACCESS_KEY", "")
    CONFIG["endpoint"] = os.getenv(
        "VOLCENGINE_ENDPOINT", "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"
    )
    CONFIG["api_key"] = os.getenv("API_KEY", "")

    model_mapping_str = os.getenv("MODEL_MAPPING", "")
    if model_mapping_str:
        try:
            CONFIG["model_mapping"] = json.loads(model_mapping_str)
        except json.JSONDecodeError:
            CONFIG["model_mapping"] = DEFAULT_MODEL_MAPPING.copy()
    else:
        CONFIG["model_mapping"] = DEFAULT_MODEL_MAPPING.copy()


async def verify_api_key(authorization: Optional[str] = Header(None)):
    """Verify API key if configured."""
    if not CONFIG["api_key"]:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    key = authorization.replace("Bearer ", "").strip()
    if key != CONFIG["api_key"]:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    language: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    _: None = Depends(verify_api_key),
):
    """OpenAI-compatible transcription endpoint (Whisper API)."""
    if not CONFIG["app_key"] or not CONFIG["access_key"]:
        raise HTTPException(status_code=500, detail="Server not configured")

    try:
        audio_data = await file.read()
        text = await transcribe_audio(audio_data, language or "zh")

        if response_format == "text":
            return text
        elif response_format == "verbose_json":
            return {"task": "transcribe", "language": language or "zh", "text": text, "segments": []}
        else:  # json
            return {"text": text}

    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models(_: None = Depends(verify_api_key)):
    """List available models."""
    models = [{"id": m, "object": "model", "owned_by": "volcengine"} for m in CONFIG["model_mapping"]]
    return {"object": "list", "data": models}


@app.on_event("startup")
async def startup_event():
    """Load config on startup."""
    if not CONFIG["app_key"]:
        load_config_from_env()
    logger.info(f"Model mappings: {list(CONFIG['model_mapping'].keys())}")


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="OpenAI-compatible ASR server")
    parser.add_argument("--app-key", default=os.getenv("VOLCENGINE_APP_KEY", ""), help="App Key")
    parser.add_argument("--access-key", default=os.getenv("VOLCENGINE_ACCESS_KEY", ""), help="Access Key")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--endpoint", default=CONFIG["endpoint"], help="WebSocket endpoint")
    parser.add_argument("--api-key", default=os.getenv("API_KEY", ""), help="Optional API key")
    args = parser.parse_args()

    CONFIG["app_key"] = args.app_key
    CONFIG["access_key"] = args.access_key
    CONFIG["endpoint"] = args.endpoint
    CONFIG["api_key"] = args.api_key
    CONFIG["model_mapping"] = DEFAULT_MODEL_MAPPING.copy()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
