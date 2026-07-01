import argparse
import asyncio
from contextlib import asynccontextmanager
from functools import lru_cache
import logging
import os
import platform
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from modules.v2.concurrent import ConcurrentInferenceParams, ConcurrentVoiceConversionService


logger = logging.getLogger(__name__)


SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".opus", ".ogg"}
SUPPORTED_OUTPUT_SUFFIXES = SUPPORTED_AUDIO_SUFFIXES
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
OUTPUT_MEDIA_TYPES = {
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".opus": "audio/ogg",
    ".ogg": "audio/ogg",
}
CLOUDFLARED_PUBLIC_URL_RE = re.compile(r"https://[-a-zA-Z0-9.]+trycloudflare\.com")
CLOUDFLARED_PATH = Path(".tools") / "cloudflared"
CLOUDFLARED_PROTOCOL = "http2"
CLOUDFLARED_MAX_ATTEMPTS = 3
CLOUDFLARED_RETRY_DELAY_SEC = 5.0
MODEL_DIR = Path("models")
AR_CHECKPOINT_PATH = MODEL_DIR / "seed-vc-v2" / "ar_base.pth"
CFM_CHECKPOINT_PATH = MODEL_DIR / "seed-vc-v2" / "cfm_small.pth"
CONTENT_EXTRACTOR_NARROW_CHECKPOINT_PATH = (
    MODEL_DIR / "astral-quantization" / "bsq32" / "bsq32_light.pth"
)
CONTENT_EXTRACTOR_WIDE_CHECKPOINT_PATH = (
    MODEL_DIR / "astral-quantization" / "bsq2048" / "bsq2048_light.pth"
)
STYLE_ENCODER_CHECKPOINT_PATH = MODEL_DIR / "campplus" / "campplus_cn_common.bin"
WARMUP_SOURCE_SEC = 20.0
WARMUP_TARGET_SEC = 20.0
WARMUP_AMPLITUDE = 0.02
CFM_BUCKET_WARMUP_DIFFUSION_STEPS = 1
FIXED_AR_SLOTS = 4


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


app = FastAPI(title="Seed-VC V2 Concurrent API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "service_ready": service is not None}


@app.get("/metrics")
async def metrics():
    if service is None:
        raise HTTPException(status_code=503, detail="service is not initialized")
    return service.metrics()


@app.post("/v2/convert")
async def convert(
    source_audio_file: UploadFile = File(..., description="Source audio file"),
    target_audio_file: UploadFile = File(..., description="Reference audio file"),
    output_format: str = Form("wav", description="Output format: wav, flac, mp3, m4a, opus, or ogg"),
    diffusion_steps: int = Form(30),
    length_adjust: float = Form(1.0),
    intelligibility_cfg_rate: float = Form(0.7),
    similarity_cfg_rate: float = Form(0.7),
    top_p: float = Form(0.7),
    temperature: float = Form(0.7),
    repetition_penalty: float = Form(1.5),
    convert_style: bool = Form(False),
    anonymization_only: bool = Form(False),
):
    if service is None:
        raise HTTPException(status_code=503, detail="service is not initialized")

    request_started_at = time.perf_counter()
    source_suffix = validate_input_file(source_audio_file, "source_audio_file")
    target_suffix = validate_input_file(target_audio_file, "target_audio_file")
    output_suffix = validate_output_format(output_format)
    if output_suffix in FFMPEG_OUTPUT_SUFFIXES and resolve_ffmpeg_executable() is None:
        raise HTTPException(
            status_code=500,
            detail=f"ffmpeg is required to write {output_suffix} output. Install system ffmpeg or imageio-ffmpeg.",
        )

    params = ConcurrentInferenceParams(
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        intelligibility_cfg_rate=intelligibility_cfg_rate,
        similarity_cfg_rate=similarity_cfg_rate,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        convert_style=convert_style,
        anonymization_only=anonymization_only,
    )

    temp_dir_path = Path(tempfile.mkdtemp(prefix="seed_vc_convert_"))
    try:
        source_path = temp_dir_path / f"source{source_suffix}"
        target_path = temp_dir_path / f"target{target_suffix}"
        upload_started_at = time.perf_counter()
        await save_upload_file(source_audio_file, source_path)
        await save_upload_file(target_audio_file, target_path)
        logger.info(
            "stage=upload_saved upload_sec=%.3f source_bytes=%s target_bytes=%s output_format=%s",
            time.perf_counter() - upload_started_at,
            source_path.stat().st_size,
            target_path.stat().st_size,
            output_suffix,
        )

        try:
            request_id, sample_rate, waveform = await service.convert(
                str(source_path),
                str(target_path),
                params,
            )
        except Exception as exc:
            logger.exception("stage=convert_failed error_type=%s", type(exc).__name__)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        filename = f"{request_id}{output_suffix}"
        output_path = temp_dir_path / filename
        encode_started_at = time.perf_counter()
        await asyncio.to_thread(write_audio_file, output_path, waveform, sample_rate, output_suffix)
        logger.info(
            "request=%s stage=output_encoded encode_sec=%.3f output_bytes=%s total_api_sec=%.3f",
            request_id,
            time.perf_counter() - encode_started_at,
            output_path.stat().st_size,
            time.perf_counter() - request_started_at,
        )
    except HTTPException:
        shutil.rmtree(temp_dir_path, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(temp_dir_path, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return FileResponse(
        output_path,
        media_type=OUTPUT_MEDIA_TYPES[output_suffix],
        filename=filename,
        background=BackgroundTask(shutil.rmtree, temp_dir_path, ignore_errors=True),
    )


def validate_input_file(upload_file: UploadFile, field_name: str) -> str:
    suffix = Path(upload_file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_AUDIO_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_SUFFIXES))
        actual = suffix or "<missing>"
        raise HTTPException(
            status_code=400,
            detail=f"unsupported {field_name} format: {actual}. Supported formats: {supported}",
        )
    return suffix


async def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output_file:
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break
            output_file.write(chunk)
    await upload_file.close()


def validate_output_format(output_format: str) -> str:
    suffix = output_format.strip().lower()
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    if suffix not in SUPPORTED_OUTPUT_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_OUTPUT_SUFFIXES))
        actual = output_format or "<missing>"
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


class ColabTunnel:
    def __init__(self, cloudflared_path: str, port: int):
        self.cloudflared_path = cloudflared_path
        self.port = port
        self.stop_event = threading.Event()
        self.process_lock = threading.Lock()
        self.process: Optional[subprocess.Popen] = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "ColabTunnel":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.stop_event.set()
        with self.process_lock:
            process = self.process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    def _run(self) -> None:
        for attempt in range(1, CLOUDFLARED_MAX_ATTEMPTS + 1):
            if self.stop_event.is_set():
                return

            process = self._start_process()
            with self.process_lock:
                self.process = process

            saw_public_url = self._stream_output(process)
            return_code = process.poll()
            if self.stop_event.is_set():
                return
            if return_code in (None, 0):
                return

            print(f"[cloudflared] tunnel exited with code {return_code}", flush=True)
            if attempt >= CLOUDFLARED_MAX_ATTEMPTS:
                print(
                    "[cloudflared] failed to start public tunnel after "
                    f"{CLOUDFLARED_MAX_ATTEMPTS} attempts. Local API is still running.",
                    flush=True,
                )
                return

            reason = "after public URL was issued" if saw_public_url else "before public URL was issued"
            print(
                f"[cloudflared] restarting tunnel attempt {attempt + 1}/{CLOUDFLARED_MAX_ATTEMPTS} "
                f"({reason})...",
                flush=True,
            )
            self.stop_event.wait(CLOUDFLARED_RETRY_DELAY_SEC)

    def _start_process(self) -> subprocess.Popen:
        return subprocess.Popen(
            [
                self.cloudflared_path,
                "tunnel",
                "--protocol",
                CLOUDFLARED_PROTOCOL,
                "--url",
                f"http://127.0.0.1:{self.port}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    @staticmethod
    def _stream_output(process: subprocess.Popen) -> bool:
        if process.stdout is None:
            return False

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

        return public_url_printed


def start_colab_tunnel(port: int) -> ColabTunnel:
    cloudflared_path = resolve_cloudflared_executable(auto_download=True)
    if cloudflared_path is None:
        raise RuntimeError("cloudflared is required for --colab mode")

    print(f"Starting Cloudflare tunnel for Colab public access (protocol={CLOUDFLARED_PROTOCOL})...", flush=True)
    return ColabTunnel(cloudflared_path, port).start()


def stop_colab_tunnel(tunnel: Optional[ColabTunnel]) -> None:
    if tunnel is None:
        return
    tunnel.stop()


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
    bigvgan_cuda_kernel_enabled = should_enable_bigvgan_cuda_kernel(device)
    cfg.vocoder.use_cuda_kernel = bigvgan_cuda_kernel_enabled
    try:
        vc_wrapper = instantiate_vc_wrapper(instantiate, cfg)
    except Exception as exc:
        if not bigvgan_cuda_kernel_enabled:
            raise
        logger.warning(
            "stage=bigvgan_cuda_kernel_fallback error_type=%s error=%s",
            type(exc).__name__,
            exc,
        )
        cfg.vocoder.use_cuda_kernel = False
        vc_wrapper = instantiate_vc_wrapper(instantiate, cfg)

    logger.info("stage=bigvgan_vocoder_loaded use_cuda_kernel=%s", bool(cfg.vocoder.use_cuda_kernel))
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
        max_batch_size=FIXED_AR_SLOTS,
        max_seq_len=args.ar_max_seq_len,
        dtype=dtype,
        device=device,
    )
    if args.ar_slots != FIXED_AR_SLOTS:
        logger.warning(
            "stage=ar_slots_fixed requested=%s using=%s",
            args.ar_slots,
            FIXED_AR_SLOTS,
        )
    if args.compile_ar or args.compile_ar_cudagraphs or args.compile_cfm or args.compile_cfm_cudagraphs:
        configure_torch_compile()
    if args.compile_cfm or args.compile_cfm_cudagraphs:
        vc_wrapper.compile_cfm(
            use_cudagraphs=args.compile_cfm_cudagraphs,
            mode=args.compile_cfm_mode,
        )
    return vc_wrapper


def instantiate_vc_wrapper(instantiate, cfg):
    previous_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    try:
        return instantiate(cfg)
    finally:
        if previous_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch_list


def should_enable_bigvgan_cuda_kernel(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        logger.info("stage=bigvgan_cuda_kernel_skipped reason=non_cuda_device device=%s", device)
        return False

    missing_tools = []
    if resolve_nvcc_executable() is None:
        missing_tools.append("nvcc")
    if shutil.which("ninja") is None:
        missing_tools.append("ninja")
    if missing_tools:
        logger.warning(
            "stage=bigvgan_cuda_kernel_skipped reason=missing_build_tools tools=%s",
            ",".join(missing_tools),
        )
        return False

    major, minor = torch.cuda.get_device_capability(device)
    logger.info("stage=bigvgan_cuda_kernel_enabled capability=sm_%s%s", major, minor)
    return True


def resolve_nvcc_executable() -> Optional[str]:
    nvcc = shutil.which("nvcc")
    if nvcc is not None:
        return nvcc
    try:
        from torch.utils import cpp_extension
    except Exception:
        return None
    cuda_home = cpp_extension.CUDA_HOME
    if cuda_home is None:
        return None
    candidate = Path(cuda_home) / "bin" / "nvcc"
    if candidate.is_file():
        return str(candidate)
    return None


def configure_torch_compile() -> None:
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    if hasattr(torch._inductor.config, "fx_graph_cache"):
        torch._inductor.config.fx_graph_cache = True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seed-VC V2 concurrent API server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--ar-slots", type=int, default=4)
    parser.add_argument("--ar-max-seq-len", type=int, default=4096)
    parser.add_argument("--cfm-max-concurrent", type=int, default=1)
    parser.add_argument("--timbre-cache-size", type=int, default=20)
    parser.add_argument("--source-cache-size", type=int, default=64)
    parser.add_argument("--cfm-batch-max-size", type=int, default=4)
    parser.add_argument("--cfm-batch-wait-sec", type=float, default=0.18)
    parser.add_argument(
        "--cfm-inline",
        action="store_true",
        help="Experimental: run CFM on the scheduler event-loop thread instead of ThreadPoolExecutor",
    )
    parser.add_argument("--feature-max-concurrent", type=int, default=1)
    parser.add_argument(
        "--ar-batch-wait-sec",
        type=float,
        default=0.0,
        help="Wait briefly for in-flight feature extraction before AR decode batching",
    )
    parser.add_argument(
        "--ar-yield-every-steps",
        type=int,
        default=1,
        help="Yield the asyncio loop every N AR decode steps; 1 preserves the original scheduler behavior",
    )
    parser.add_argument(
        "--ar-eos-check-interval",
        type=int,
        default=1,
        help="Check AR EOS every N decode steps; 1 preserves exact per-token EOS checking",
    )
    parser.add_argument("--compile-ar", action="store_true")
    parser.add_argument(
        "--compile-ar-cudagraphs",
        action="store_true",
        help="Experimental: use torch.compile reduce-overhead/CUDA Graphs for AR decode",
    )
    parser.add_argument(
        "--compile-ar-sampling",
        action="store_true",
        help="Experimental: compile AR sampling; disabled by default due dynamic-shape recompilation risk",
    )
    parser.add_argument("--compile-cfm", action="store_true")
    parser.add_argument(
        "--compile-cfm-cudagraphs",
        action="store_true",
        help="Experimental: use torch.compile reduce-overhead/CUDA Graphs for CFM",
    )
    parser.add_argument(
        "--compile-cfm-mode",
        choices=("default", "reduce-overhead", "max-autotune-no-cudagraphs", "max-autotune"),
        default=None,
        help="Optional torch.compile mode for CFM transformer; omitted preserves the legacy no-cudagraph path",
    )
    parser.add_argument("--enable-profiling", action="store_true", help="Enable detailed profiling logs and CUDA sync timing")
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
        ar_slots=FIXED_AR_SLOTS,
        ar_max_seq_len=args.ar_max_seq_len,
        timbre_cache_size=args.timbre_cache_size,
        source_cache_size=args.source_cache_size,
        cfm_max_concurrent=args.cfm_max_concurrent,
        cfm_batch_max_size=args.cfm_batch_max_size,
        cfm_batch_wait_sec=args.cfm_batch_wait_sec,
        cfm_inline=args.cfm_inline,
        feature_max_concurrent=args.feature_max_concurrent,
        ar_batch_wait_sec=args.ar_batch_wait_sec,
        ar_yield_every_steps=args.ar_yield_every_steps,
        ar_eos_check_interval=args.ar_eos_check_interval,
        enable_profiling=args.enable_profiling,
        compile_ar=args.compile_ar or args.compile_ar_cudagraphs,
        compile_ar_cudagraphs=args.compile_ar_cudagraphs,
        compile_ar_sampling=args.compile_ar_sampling,
    )
    warmed_ar_batches = await service.warmup_ar_decode()
    if warmed_ar_batches:
        logger.info(
            "stage=warmup_ar_decode_batches_done batches=%s",
            ",".join(str(batch_size) for batch_size in warmed_ar_batches),
        )
    await service.start()
    try:
        await run_startup_warmup(service)
    except Exception:
        await service.stop()
        raise


async def run_startup_warmup(started_service: ConcurrentVoiceConversionService) -> None:
    warmup_dir_path = Path(tempfile.mkdtemp(prefix="seed_vc_warmup_"))
    warmup_started_at = time.perf_counter()
    try:
        source_path = warmup_dir_path / "warmup_source.wav"
        target_path = warmup_dir_path / "warmup_target.wav"
        sample_rate = int(started_service.vc_wrapper.sr)
        write_warmup_audio(source_path, sample_rate, WARMUP_SOURCE_SEC, base_freq=220.0)
        write_warmup_audio(target_path, sample_rate, WARMUP_TARGET_SEC, base_freq=330.0)
        logger.info(
            "stage=warmup_start source_sec=%.1f target_sec=%.1f sample_rate=%s",
            WARMUP_SOURCE_SEC,
            WARMUP_TARGET_SEC,
            sample_rate,
        )
        request_id, output_sample_rate, waveform = await started_service.convert(
            str(source_path),
            str(target_path),
            ConcurrentInferenceParams(),
        )
        logger.info(
            "request=%s stage=warmup_done elapsed_sec=%.3f output_sample_rate=%s output_samples=%s",
            request_id,
            time.perf_counter() - warmup_started_at,
            output_sample_rate,
            len(waveform),
        )
        warmed_buckets = await started_service.warmup_cfm_compile_buckets(
            str(target_path),
            ConcurrentInferenceParams(diffusion_steps=CFM_BUCKET_WARMUP_DIFFUSION_STEPS),
        )
        if warmed_buckets:
            logger.info(
                "stage=warmup_cfm_buckets_done buckets=%s",
                ",".join(str(bucket_len) for bucket_len in warmed_buckets),
            )
        started_service.timbre_cache.clear()
        started_service.source_cache.clear()
    finally:
        shutil.rmtree(warmup_dir_path, ignore_errors=True)


def write_warmup_audio(path: Path, sample_rate: int, duration_sec: float, base_freq: float) -> None:
    sample_count = max(1, int(sample_rate * duration_sec))
    timeline = np.arange(sample_count, dtype=np.float32) / sample_rate
    waveform = WARMUP_AMPLITUDE * (
        np.sin(2 * np.pi * base_freq * timeline)
        + 0.35 * np.sin(2 * np.pi * base_freq * 1.5 * timeline)
    )
    sf.write(path, waveform.astype(np.float32), sample_rate)


def main():
    global server_args
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = build_arg_parser().parse_args()
    server_args = args
    tunnel_process = start_colab_tunnel(args.port) if args.colab else None
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        stop_colab_tunnel(tunnel_process)


if __name__ == "__main__":
    main()
