# Seed-VC V2 API Service

This repository has been trimmed to the V2 inference path only. WebUI, V1,
training, evaluation, and baseline code have been removed.

## Runtime

- API server: `api_v2_concurrent.py`
- Optional single-request CLI: `inference_v2.py`
- V2 config: `configs/v2/vc_wrapper.yaml`
- Core modules: `modules/v2`, `modules/astral_quantization`, `modules/campplus`,
  `modules/bigvgan`

## Install

```bash
pip install -r requirements.txt
```

Choose the appropriate PyTorch wheel for your CUDA/runtime environment if the
default pinned wheel is not suitable.

## Prepare Models

The API runtime is offline-only and does not download model files. Prepare the
local model directory before starting the service:

```bash
python scripts/download_models.py --model-dir models
```

Expected layout:

```text
models/
  seed-vc-v2/cfm_small.pth
  seed-vc-v2/ar_base.pth
  astral-quantization/bsq32/bsq32_light.pth
  astral-quantization/bsq2048/bsq2048_light.pth
  campplus/campplus_cn_common.bin
  bigvgan_v2_22khz_80band_256x/config.json
  bigvgan_v2_22khz_80band_256x/bigvgan_generator.pt
  hubert-large-ll60k/config.json
  hubert-large-ll60k/preprocessor_config.json
  hubert-large-ll60k/pytorch_model.bin
```

## Start API

```bash
python api_v2_concurrent.py --ar-slots 4 --cfm-max-concurrent 1 --port 8000
```

`--ar-slots` controls fixed-slot AR concurrency. `--cfm-max-concurrent` limits
CFM/Vocoder concurrency to avoid GPU OOM; start with `1`, then increase after
pressure testing.

For Colab testing, add `--colab` to start a Cloudflare tunnel and print a
public API URL:

```bash
python api_v2_concurrent.py --colab --ar-slots 2 --cfm-max-concurrent 1 --port 8000
```

## Convert

```bash
curl -X POST http://127.0.0.1:8000/v2/convert \
  -F "source_audio_file=@examples/source/source_s1.wav" \
  -F "target_audio_file=@examples/reference/s1p1.wav" \
  -F "output_path=outputs/api_v2/result.wav" \
  -F "convert_style=false"
```

`source_audio_file` and `target_audio_file` are uploaded as multipart files.
Supported input and output suffixes are `.wav`, `.flac`, `.mp3`, `.m4a`,
`.opus`, and `.ogg`.

Useful endpoints:

- `GET /health`
- `GET /metrics`
- `POST /v2/convert`
