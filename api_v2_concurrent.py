import argparse
import asyncio
from contextlib import asynccontextmanager
from functools import lru_cache
import platform
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from modules.v2.concurrent import ConcurrentInferenceParams, ConcurrentVoiceConversionService


SUPPORTED_OUTPUT_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".opus", ".ogg"}
SOUNDFILE_OUTPUT_FORMATS = {
    ".wav": "WAV",
    ".flac": "FLAC",
    ".ogg": "OGG",
}
FFMPEG_OUTPUT_SUFFIXES = {".mp3", ".m4a", ".opus"}
FFMPEG_OUTPUT_ARGS = {
    ".mp3": ["-codec:a", "libmp3lame", "-b:a", "320k"],
    ".m4a": ["-codec:a", "aac", "-b:a", "192k"],
    ".opus": ["-codec:a", "libopus", "-b:a", "128k"],
}
CLOUDFLARED_PUBLIC_URL_RE = re.compile(r"https://[-a-zA-Z0-9.]+trycloudflare\.com")
CLOUDFLARED_PATH = Path(".tools") / "cloudflared"
MODEL_DIR = Path("models")
DEFAULT_OUTPUT_DIR = Path("outputs/api_v2")
AR_CHECKPOINT_PATH = MODEL_DIR / "seed-vc-v2" / "ar_base.pth"
CFM_CHECKPOINT_PATH = MODEL_DIR / "seed-vc-v2" / "cfm_small.pth"
CONTENT_EXTRACTOR_NARROW_CHECKPOINT_PATH = (
    MODEL_DIR / "astral-quantization" / "bsq32" / "bsq32_light.pth"
)
CONTENT_EXTRACTOR_WIDE_CHECKPOINT_PATH = (
    MODEL_DIR / "astral-quantization" / "bsq2048" / "bsq2048_light.pth"
)
STYLE_ENCODER_CHECKPOINT_PATH = MODEL_DIR / "campplus" / "campplus_cn_common.bin"


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


class ConvertRequest(BaseModel):
    source_audio_path: str = Field(..., description="Source audio file path on the server")
    target_audio_path: str = Field(..., description="Reference audio file path on the server")
    output_path: Optional[str] = Field(
        None,
        description="Optional output path ending with .wav, .flac, .mp3, .m4a, .opus, or .ogg",
    )
    diffusion_steps: int = 30
    length_adjust: float = 1.0
    intelligibility_cfg_rate: float = 0.7
    similarity_cfg_rate: float = 0.7
    top_p: float = 0.7
    temperature: float = 0.7
    repetition_penalty: float = 1.5
    convert_style: bool = False
    anonymization_only: bool = False


service: Optional[ConcurrentVoiceConversionService] = None
server_args = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if server_args is not None:
        await initialize_service(server_args)
    try:
        yield
    finally:
        if service is not None:
            await service.stop()


class ConvertResponse(BaseModel):
    request_id: str
    output_path: str
    sample_rate: int
    duration_sec: float
    elapsed_sec: float


app = FastAPI(title="Seed-VC V2 Concurrent API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "service_ready": service is not None}


@app.get("/metrics")
async def metrics():
    if service is None:
        raise HTTPException(status_code=503, detail="service is not initialized")
    return service.metrics()


@app.post("/v2/convert", response_model=ConvertResponse)
async def convert(request: ConvertRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="service is not initialized")

    source_path = Path(request.source_audio_path).expanduser()
    target_path = Path(request.target_audio_path).expanduser()
    if not source_path.exists():
        raise HTTPException(status_code=400, detail=f"source file does not exist: {source_path}")
    if not target_path.exists():
        raise HTTPException(status_code=400, detail=f"target file does not exist: {target_path}")

    output_path = Path(request.output_path).expanduser() if request.output_path else None
    output_suffix = validate_output_path(output_path) if output_path is not None else ".wav"
    if output_suffix in FFMPEG_OUTPUT_SUFFIXES and resolve_ffmpeg_executable() is None:
        raise HTTPException(
            status_code=500,
            detail=f"ffmpeg is required to write {output_suffix} output. Install system ffmpeg or imageio-ffmpeg.",
        )

    params = ConcurrentInferenceParams(
        diffusion_steps=request.diffusion_steps,
        length_adjust=request.length_adjust,
        intelligibility_cfg_rate=request.intelligibility_cfg_rate,
        similarity_cfg_rate=request.similarity_cfg_rate,
        top_p=request.top_p,
        temperature=request.temperature,
        repetition_penalty=request.repetition_penalty,
        convert_style=request.convert_style,
        anonymization_only=request.anonymization_only,
    )

    started_at = time.time()
    try:
        request_id, sample_rate, waveform = await service.convert(
            str(source_path),
            str(target_path),
            params,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / f"{request_id}.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(write_audio_file, output_path, waveform, sample_rate, output_suffix)
    elapsed = time.time() - started_at
    return ConvertResponse(
        request_id=request_id,
        output_path=str(output_path.resolve()),
        sample_rate=sample_rate,
        duration_sec=float(len(waveform) / sample_rate),
        elapsed_sec=elapsed,
    )


def validate_output_path(output_path: Path) -> str:
    suffix = output_path.suffix.lower()
    if suffix not in SUPPORTED_OUTPUT_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_OUTPUT_SUFFIXES))
        actual = suffix or "<missing>"
        raise HTTPException(
            status_code=400,
            detail=f"unsupported output format: {actual}. Supported formats: {supported}",
        )
    return suffix


def write_audio_file(output_path: Path, waveform: np.ndarray, sample_rate: int, suffix: str) -> None:
    if suffix in SOUNDFILE_OUTPUT_FORMATS:
        sf.write(str(output_path), waveform, sample_rate, format=SOUNDFILE_OUTPUT_FORMATS[suffix])
        return
    write_audio_file_with_ffmpeg(output_path, waveform, sample_rate, suffix)


def write_audio_file_with_ffmpeg(output_path: Path, waveform: np.ndarray, sample_rate: int, suffix: str) -> None:
    ffmpeg_path = resolve_ffmpeg_executable()
    if ffmpeg_path is None:
        raise RuntimeError(f"ffmpeg is required to write {suffix} output. Install system ffmpeg or imageio-ffmpeg.")

    audio, channels = prepare_audio_for_ffmpeg(waveform)
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-i",
        "pipe:0",
        *FFMPEG_OUTPUT_ARGS[suffix],
        str(output_path),
    ]
    result = subprocess.run(command, input=audio.tobytes(), capture_output=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to write {suffix} output: {stderr}")


@lru_cache(maxsize=1)
def resolve_ffmpeg_executable() -> Optional[str]:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg
    except ImportError:
        return None

    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def prepare_audio_for_ffmpeg(waveform: np.ndarray) -> tuple[np.ndarray, int]:
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim == 1:
        channels = 1
    elif audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]:
            audio = audio.T
        channels = audio.shape[1]
    else:
        raise ValueError(f"unsupported waveform shape for audio export: {audio.shape}")

    audio = np.nan_to_num(audio, copy=False)
    audio = np.clip(audio, -1.0, 1.0)
    return np.ascontiguousarray(audio), channels


def start_colab_tunnel(port: int) -> subprocess.Popen:
    cloudflared_path = resolve_cloudflared_executable(auto_download=True)
    if cloudflared_path is None:
        raise RuntimeError("cloudflared is required for --colab mode")

    print("Starting Cloudflare tunnel for Colab public access...", flush=True)
    process = subprocess.Popen(
        [
            cloudflared_path,
            "tunnel",
            "--url",
            f"http://127.0.0.1:{port}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(target=stream_colab_tunnel_output, args=(process,), daemon=True).start()
    return process


def stop_colab_tunnel(process: Optional[subprocess.Popen]) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def stream_colab_tunnel_output(process: subprocess.Popen) -> None:
    if process.stdout is None:
        return

    public_url_printed = False
    for line in process.stdout:
        line = line.strip()
        match = CLOUDFLARED_PUBLIC_URL_RE.search(line)
        if match and not public_url_printed:
            public_url = match.group(0)
            public_url_printed = True
            print("", flush=True)
            print(f"Colab public API URL: {public_url}", flush=True)
            print(f"Colab public API docs: {public_url}/docs", flush=True)
            print("", flush=True)
        elif "error" in line.lower() or "failed" in line.lower():
            print(f"[cloudflared] {line}", flush=True)

    return_code = process.poll()
    if return_code not in (None, 0):
        print(f"[cloudflared] tunnel exited with code {return_code}", flush=True)


def resolve_cloudflared_executable(auto_download: bool = False) -> Optional[str]:
    system_cloudflared = shutil.which("cloudflared")
    if system_cloudflared:
        return system_cloudflared

    if CLOUDFLARED_PATH.is_file():
        return str(CLOUDFLARED_PATH.resolve())

    if not auto_download:
        return None

    return download_cloudflared()


def download_cloudflared() -> str:
    url = cloudflared_download_url()
    CLOUDFLARED_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = CLOUDFLARED_PATH.with_suffix(".tmp")
    print(f"Downloading cloudflared from {url}...", flush=True)

    if shutil.which("curl"):
        command = [
            "curl",
            "-L",
            "--fail",
            "--show-error",
            "--silent",
            url,
            "-o",
            str(tmp_path),
        ]
    elif shutil.which("wget"):
        command = ["wget", "-q", url, "-O", str(tmp_path)]
    else:
        raise RuntimeError("curl or wget is required to download cloudflared in --colab mode")

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"failed to download cloudflared: {message}")

    tmp_path.chmod(0o755)
    tmp_path.replace(CLOUDFLARED_PATH)
    return str(CLOUDFLARED_PATH.resolve())


def cloudflared_download_url() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux" and machine in {"x86_64", "amd64"}:
        return "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    if system == "linux" and machine in {"aarch64", "arm64"}:
        return "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64"
    raise RuntimeError(f"automatic cloudflared download is not supported on {system}/{machine}")


def load_v2_models(args, device: torch.device, dtype: torch.dtype):
    from hydra.utils import instantiate
    from omegaconf import DictConfig

    cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
    vc_wrapper = instantiate(cfg)
    vc_wrapper.load_checkpoints(
        ar_checkpoint_path=AR_CHECKPOINT_PATH,
        cfm_checkpoint_path=CFM_CHECKPOINT_PATH,
        content_extractor_narrow_checkpoint_path=CONTENT_EXTRACTOR_NARROW_CHECKPOINT_PATH,
        content_extractor_wide_checkpoint_path=CONTENT_EXTRACTOR_WIDE_CHECKPOINT_PATH,
        style_encoder_checkpoint_path=STYLE_ENCODER_CHECKPOINT_PATH,
    )
    vc_wrapper.to(device)
    vc_wrapper.eval()
    vc_wrapper.setup_ar_caches(
        max_batch_size=args.ar_slots,
        max_seq_len=args.ar_max_seq_len,
        dtype=dtype,
        device=device,
    )
    if args.compile_cfm:
        vc_wrapper.compile_cfm()
    return vc_wrapper


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seed-VC V2 concurrent API server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--ar-slots", type=int, default=4)
    parser.add_argument("--ar-max-seq-len", type=int, default=4096)
    parser.add_argument("--cfm-max-concurrent", type=int, default=1)
    parser.add_argument("--timbre-cache-size", type=int, default=20)
    parser.add_argument("--compile-cfm", action="store_true")
    parser.add_argument("--colab", action="store_true", help="Start a Cloudflare tunnel and print a public URL")
    return parser


async def initialize_service(args) -> None:
    global service
    device = select_device()
    dtype = parse_dtype(args.dtype)
    vc_wrapper = await asyncio.to_thread(load_v2_models, args, device, dtype)
    service = ConcurrentVoiceConversionService(
        vc_wrapper,
        device=device,
        dtype=dtype,
        ar_slots=args.ar_slots,
        ar_max_seq_len=args.ar_max_seq_len,
        timbre_cache_size=args.timbre_cache_size,
        cfm_max_concurrent=args.cfm_max_concurrent,
    )
    await service.start()


def main():
    global server_args
    args = build_arg_parser().parse_args()
    server_args = args
    tunnel_process = start_colab_tunnel(args.port) if args.colab else None
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        stop_colab_tunnel(tunnel_process)


if __name__ == "__main__":
    main()
