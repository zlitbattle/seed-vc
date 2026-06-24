import asyncio
import hashlib
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import torch


logger = logging.getLogger(__name__)


@dataclass
class ConcurrentInferenceParams:
    diffusion_steps: int = 30
    length_adjust: float = 1.0
    intelligibility_cfg_rate: float = 0.7
    similarity_cfg_rate: float = 0.7
    top_p: float = 0.7
    temperature: float = 0.7
    repetition_penalty: float = 1.5
    convert_style: bool = False
    anonymization_only: bool = False


@dataclass
class TimbreFeatures:
    cache_key: str
    target_audio_path: str
    target_mel: torch.Tensor
    target_mel_len: int
    target_content_indices: torch.Tensor
    target_narrow_reduced: torch.Tensor
    target_style: torch.Tensor
    prompt_condition: torch.Tensor
    created_at: float = field(default_factory=time.time)


@dataclass
class SourceFeatures:
    source_audio_path: str
    source_mel_len: int
    source_content_indices: torch.Tensor
    source_narrow_reduced: Optional[torch.Tensor] = None


@dataclass
class ARGenerateRequest:
    request_id: str
    prompt_text: torch.Tensor
    prompt_target: torch.Tensor
    params: ConcurrentInferenceParams
    future: Optional[asyncio.Future] = None
    slot_id: Optional[int] = None
    is_prefilled: bool = False
    next_input_pos: Optional[torch.Tensor] = None
    next_kv_pos: Optional[torch.Tensor] = None
    last_emb: Optional[torch.Tensor] = None
    generated_tokens: List[torch.Tensor] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)


@dataclass
class CFMJob:
    request_id: str
    params: ConcurrentInferenceParams
    source_features: SourceFeatures
    timbre_features: TimbreFeatures
    ar_outputs: Optional[List[torch.Tensor]] = None
    future: Optional[asyncio.Future] = None
    submitted_at: float = field(default_factory=time.perf_counter)


class TimbreFeatureCache:
    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self._items: "OrderedDict[str, TimbreFeatures]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[TimbreFeatures]:
        item = self._items.get(key)
        if item is None:
            self.misses += 1
            return None
        self._items.move_to_end(key)
        self.hits += 1
        return item

    def put(self, key: str, features: TimbreFeatures) -> None:
        self._items[key] = features
        self._items.move_to_end(key)
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)

    def metrics(self) -> Dict[str, float]:
        total = self.hits + self.misses
        hit_rate = self.hits / total if total else 0.0
        return {
            "size": len(self._items),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }


class ARScheduler:
    def __init__(
        self,
        ar_wrapper,
        max_slots: int,
        max_seq_len: int = 4096,
        min_tokens_before_eos: int = 10,
        max_new_tokens: int = 4000,
    ):
        self.ar_wrapper = ar_wrapper
        self.max_slots = max_slots
        self.max_seq_len = max_seq_len
        self.min_tokens_before_eos = min_tokens_before_eos
        self.max_new_tokens = max_new_tokens
        self.waiting_queue: "asyncio.Queue[ARGenerateRequest]" = asyncio.Queue()
        self.active: List[ARGenerateRequest] = []
        self.free_slots = list(range(max_slots))
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self.completed = 0
        self.failed = 0

    async def start(self) -> None:
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._schedule_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def submit(self, request: ARGenerateRequest) -> torch.Tensor:
        request.future = asyncio.get_running_loop().create_future()
        await self.waiting_queue.put(request)
        return await request.future

    async def _schedule_loop(self) -> None:
        while self._running:
            try:
                await self._fill_slots()
                if not self.active:
                    request = await self.waiting_queue.get()
                    self._activate(request)

                for request in list(self.active):
                    if not request.is_prefilled:
                        self._prefill_one(request)

                decode_requests = [request for request in self.active if request.is_prefilled]
                if decode_requests:
                    self._decode_one_step(decode_requests)

                await asyncio.sleep(0)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._fail_all(exc)
                await asyncio.sleep(0.01)

    async def _fill_slots(self) -> None:
        while self.free_slots and not self.waiting_queue.empty():
            request = self.waiting_queue.get_nowait()
            self._activate(request)

    def _activate(self, request: ARGenerateRequest) -> None:
        if not self.free_slots:
            raise RuntimeError("AR slot exhausted")
        request.slot_id = self.free_slots.pop(0)
        self.active.append(request)

    def _prefill_one(self, request: ARGenerateRequest) -> None:
        assert request.slot_id is not None
        emb_seq, input_pos, kv_pos = self.ar_wrapper.build_generation_inputs(
            request.prompt_text,
            request.prompt_target,
        )
        if emb_seq.size(1) >= self.max_seq_len:
            raise RuntimeError(
                f"AR prompt is too long: {emb_seq.size(1)} >= max_seq_len {self.max_seq_len}"
            )

        eos_token = self.ar_wrapper.model.config.vocab_size - 1
        next_tokens = self.ar_wrapper.decode_one_token_ar(
            emb_seq,
            input_pos,
            kv_pos,
            slot_ids=[request.slot_id],
            suppress_tokens=[eos_token],
            top_p=request.params.top_p,
            temperature=request.params.temperature,
            repetition_penalty=request.params.repetition_penalty,
        )
        token = next_tokens[0].reshape(()).clone()
        request.generated_tokens.append(token)
        request.last_emb = self.ar_wrapper.embed_generated_token(token)
        request.next_input_pos = input_pos[-1:] + 1
        request.next_kv_pos = kv_pos[-1:] + 1
        request.is_prefilled = True

    def _decode_one_step(self, requests: Sequence[ARGenerateRequest]) -> None:
        eos_token = self.ar_wrapper.model.config.vocab_size - 1
        for request in list(requests):
            if len(request.generated_tokens) >= self.max_new_tokens:
                self._finish(request)

        live_requests = [
            request for request in requests
            if request in self.active and request.last_emb is not None and len(request.generated_tokens) < self.max_new_tokens
        ]
        if not live_requests:
            return

        groups: Dict[Tuple[float, float, float], List[ARGenerateRequest]] = {}
        for request in live_requests:
            key = (
                request.params.top_p,
                request.params.temperature,
                request.params.repetition_penalty,
            )
            groups.setdefault(key, []).append(request)

        for (top_p, temperature, repetition_penalty), group in groups.items():
            x = torch.cat([request.last_emb for request in group], dim=0)
            input_pos = torch.stack([request.next_input_pos.reshape(()) for request in group], dim=0).view(-1, 1)
            kv_pos = torch.stack([request.next_kv_pos.reshape(()) for request in group], dim=0).view(-1, 1)
            slot_ids = [request.slot_id for request in group]
            previous_tokens = [torch.stack([token.reshape(()) for token in request.generated_tokens]) for request in group]
            suppress_tokens = [
                [eos_token] if len(request.generated_tokens) < self.min_tokens_before_eos else None
                for request in group
            ]

            next_tokens = self.ar_wrapper.decode_one_token_ar_batch(
                x,
                input_pos,
                kv_pos,
                slot_ids=slot_ids,
                previous_tokens=previous_tokens,
                suppress_tokens=suppress_tokens,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )

            for request, token in zip(group, next_tokens):
                token = token.reshape(()).clone()
                reached_eos = token.item() == eos_token and len(request.generated_tokens) >= self.min_tokens_before_eos
                reached_limit = len(request.generated_tokens) + 1 >= self.max_new_tokens
                reached_cache_limit = int(request.next_kv_pos.item()) + 1 >= self.max_seq_len
                if reached_eos or reached_limit or reached_cache_limit:
                    self._finish(request)
                    continue

                request.generated_tokens.append(token)
                request.last_emb = self.ar_wrapper.embed_generated_token(token)
                request.next_input_pos = request.next_input_pos + 1
                request.next_kv_pos = request.next_kv_pos + 1

    def _finish(self, request: ARGenerateRequest) -> None:
        if request in self.active:
            self.active.remove(request)
        if request.slot_id is not None:
            self.free_slots.append(request.slot_id)
            self.free_slots.sort()
        result = torch.stack([token.reshape(()) for token in request.generated_tokens], dim=0).long().unsqueeze(0)
        if request.future is not None and not request.future.done():
            request.future.set_result(result)
        self.completed += 1

    def _fail_all(self, exc: Exception) -> None:
        for request in list(self.active):
            self._fail_request(request, exc)

    def _fail_request(self, request: ARGenerateRequest, exc: Exception) -> None:
        if request in self.active:
            self.active.remove(request)
        if request.slot_id is not None and request.slot_id not in self.free_slots:
            self.free_slots.append(request.slot_id)
            self.free_slots.sort()
        if request.future is not None and not request.future.done():
            request.future.set_exception(exc)
        self.failed += 1

    def metrics(self) -> Dict[str, int]:
        return {
            "max_slots": self.max_slots,
            "active_requests": len(self.active),
            "queue_length": self.waiting_queue.qsize(),
            "free_slots": len(self.free_slots),
            "completed": self.completed,
            "failed": self.failed,
        }


class CFMScheduler:
    def __init__(self, vc_wrapper, device: torch.device, dtype: torch.dtype, max_concurrent: int = 1):
        self.vc_wrapper = vc_wrapper
        self.device = device
        self.dtype = dtype
        self.max_concurrent = max(1, max_concurrent)
        self.queue: "asyncio.Queue[CFMJob]" = asyncio.Queue()
        self._tasks: List[asyncio.Task] = []
        self._running = False
        self.active = 0
        self.completed = 0
        self.failed = 0

    async def start(self) -> None:
        if self._tasks:
            return
        self._running = True
        for _ in range(self.max_concurrent):
            self._tasks.append(asyncio.create_task(self._worker()))

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()

    async def submit(self, job: CFMJob) -> Tuple[int, np.ndarray]:
        job.future = asyncio.get_running_loop().create_future()
        logger.info(
            "request=%s stage=cfm_queue_submit queue_size=%s active=%s max_concurrent=%s",
            job.request_id,
            self.queue.qsize(),
            self.active,
            self.max_concurrent,
        )
        await self.queue.put(job)
        return await job.future

    async def _worker(self) -> None:
        while self._running:
            job = await self.queue.get()
            self.active += 1
            queue_wait_sec = time.perf_counter() - job.submitted_at
            run_started_at = time.perf_counter()
            logger.info(
                "request=%s stage=cfm_start queue_wait_sec=%.3f active=%s queue_size=%s",
                job.request_id,
                queue_wait_sec,
                self.active,
                self.queue.qsize(),
            )
            try:
                result = await asyncio.to_thread(self._run_job, job)
                if job.future is not None and not job.future.done():
                    job.future.set_result(result)
                self.completed += 1
                logger.info(
                    "request=%s stage=cfm_done run_sec=%.3f completed=%s",
                    job.request_id,
                    time.perf_counter() - run_started_at,
                    self.completed,
                )
            except Exception as exc:
                if job.future is not None and not job.future.done():
                    job.future.set_exception(exc)
                self.failed += 1
                logger.exception(
                    "request=%s stage=cfm_failed run_sec=%.3f",
                    job.request_id,
                    time.perf_counter() - run_started_at,
                )
            finally:
                self.active -= 1

    @torch.no_grad()
    @torch.inference_mode()
    def _run_job(self, job: CFMJob) -> Tuple[int, np.ndarray]:
        if job.params.convert_style:
            return self._run_style_job(job)
        return self._run_timbre_job(job)

    def _run_timbre_job(self, job: CFMJob) -> Tuple[int, np.ndarray]:
        source = job.source_features
        timbre = job.timbre_features
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            cond, _ = self.vc_wrapper.cfm_length_regulator(
                source.source_content_indices,
                ylens=torch.LongTensor([source.source_mel_len]).to(self.device),
            )
        return self._render_condition_chunks(job, cond)

    def _run_style_job(self, job: CFMJob) -> Tuple[int, np.ndarray]:
        if not job.ar_outputs:
            raise RuntimeError("convert_style=True requires AR outputs")

        generated_wave_chunks = []
        previous_chunk = None
        processed_frames = 0
        overlap_wave_len = self.vc_wrapper.overlap_frame_len * self.vc_wrapper.hop_size

        source = job.source_features
        timbre = job.timbre_features
        for index, ar_out in enumerate(job.ar_outputs):
            is_last_chunk = index + 1 >= len(job.ar_outputs)
            ar_out_mel_len = torch.LongTensor([
                int(
                    source.source_mel_len
                    / source.source_content_indices.size(-1)
                    * ar_out.size(-1)
                    * job.params.length_adjust
                )
            ]).to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                chunk_cond, _ = self.vc_wrapper.cfm_length_regulator(ar_out, ylens=ar_out_mel_len)
                cat_condition = torch.cat([timbre.prompt_condition, chunk_cond], dim=1)
                original_len = cat_condition.size(1)
                vc_mel = self._infer_cfm(job, cat_condition, random_voice=job.params.anonymization_only)
            vc_mel = vc_mel[:, :, timbre.target_mel_len:original_len]
            vc_wave = self.vc_wrapper.vocoder(vc_mel).squeeze()[None]
            processed_frames, previous_chunk, should_break, _, full_audio = self.vc_wrapper._stream_wave_chunks(
                vc_wave,
                processed_frames,
                vc_mel,
                overlap_wave_len,
                generated_wave_chunks,
                previous_chunk,
                is_last_chunk,
                stream_output=False,
            )
            if should_break:
                return self.vc_wrapper.sr, full_audio
        return self.vc_wrapper.sr, np.concatenate(generated_wave_chunks)

    def _render_condition_chunks(self, job: CFMJob, cond: torch.Tensor) -> Tuple[int, np.ndarray]:
        timbre = job.timbre_features
        max_context_window = self.vc_wrapper.sr // self.vc_wrapper.hop_size * self.vc_wrapper.dit_max_context_len
        max_source_window = max(1, max_context_window - timbre.target_mel_len)
        overlap_wave_len = self.vc_wrapper.overlap_frame_len * self.vc_wrapper.hop_size
        generated_wave_chunks = []
        processed_frames = 0
        previous_chunk = None
        chunk_index = 0

        while processed_frames < cond.size(1):
            chunk_started_at = time.perf_counter()
            chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
            is_last_chunk = processed_frames + max_source_window >= cond.size(1)
            cat_condition = torch.cat([timbre.prompt_condition, chunk_cond], dim=1)
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                vc_mel = self._infer_cfm(job, cat_condition, random_voice=job.params.anonymization_only)
            original_len = cat_condition.size(1)
            vc_mel = vc_mel[:, :, timbre.target_mel_len:original_len]
            vc_wave = self.vc_wrapper.vocoder(vc_mel).squeeze()[None]
            processed_frames, previous_chunk, should_break, _, full_audio = self.vc_wrapper._stream_wave_chunks(
                vc_wave,
                processed_frames,
                vc_mel,
                overlap_wave_len,
                generated_wave_chunks,
                previous_chunk,
                is_last_chunk,
                stream_output=False,
            )
            if should_break:
                logger.info(
                    "request=%s stage=cfm_chunk_done chunk=%s chunk_sec=%.3f last=%s",
                    job.request_id,
                    chunk_index,
                    time.perf_counter() - chunk_started_at,
                    is_last_chunk,
                )
                return self.vc_wrapper.sr, full_audio
            logger.info(
                "request=%s stage=cfm_chunk_done chunk=%s chunk_sec=%.3f last=%s",
                job.request_id,
                chunk_index,
                time.perf_counter() - chunk_started_at,
                is_last_chunk,
            )
            chunk_index += 1

        return self.vc_wrapper.sr, np.concatenate(generated_wave_chunks)

    def _infer_cfm(self, job: CFMJob, cat_condition: torch.Tensor, random_voice: bool) -> torch.Tensor:
        original_len = cat_condition.size(1)
        if self.vc_wrapper.dit_compiled:
            cat_condition = torch.nn.functional.pad(
                cat_condition,
                (0, 0, 0, self.vc_wrapper.compile_len - cat_condition.size(1)),
                value=0,
            )
        timbre = job.timbre_features
        return self.vc_wrapper.cfm.inference(
            cat_condition,
            torch.LongTensor([original_len]).to(self.device),
            timbre.target_mel,
            timbre.target_style,
            job.params.diffusion_steps,
            inference_cfg_rate=[job.params.intelligibility_cfg_rate, job.params.similarity_cfg_rate],
            random_voice=random_voice,
        )

    def metrics(self) -> Dict[str, int]:
        return {
            "max_concurrent": self.max_concurrent,
            "active_cfm": self.active,
            "queue_length": self.queue.qsize(),
            "completed": self.completed,
            "failed": self.failed,
        }


class ConcurrentVoiceConversionService:
    def __init__(
        self,
        vc_wrapper,
        device: torch.device,
        dtype: torch.dtype,
        ar_slots: int = 4,
        ar_max_seq_len: int = 4096,
        timbre_cache_size: int = 20,
        cfm_max_concurrent: int = 1,
    ):
        self.vc_wrapper = vc_wrapper
        self.device = device
        self.dtype = dtype
        self.ar_scheduler = ARScheduler(
            vc_wrapper.ar,
            max_slots=ar_slots,
            max_seq_len=ar_max_seq_len,
        )
        self.cfm_scheduler = CFMScheduler(
            vc_wrapper,
            device=device,
            dtype=dtype,
            max_concurrent=cfm_max_concurrent,
        )
        self.timbre_cache = TimbreFeatureCache(max_size=timbre_cache_size)
        self._feature_lock = asyncio.Lock()

    async def start(self) -> None:
        await self.ar_scheduler.start()
        await self.cfm_scheduler.start()

    async def stop(self) -> None:
        await self.ar_scheduler.stop()
        await self.cfm_scheduler.stop()

    async def convert(
        self,
        source_audio_path: str,
        target_audio_path: str,
        params: ConcurrentInferenceParams,
    ) -> Tuple[str, int, np.ndarray]:
        request_id = str(uuid.uuid4())
        total_started_at = time.perf_counter()
        feature_lock_wait_started_at = time.perf_counter()
        async with self._feature_lock:
            feature_lock_wait_sec = time.perf_counter() - feature_lock_wait_started_at
            prepare_started_at = time.perf_counter()
            source_features, timbre_features = await asyncio.to_thread(
                self._prepare_features,
                request_id,
                source_audio_path,
                target_audio_path,
                params.convert_style,
            )
            logger.info(
                "request=%s stage=prepare_done lock_wait_sec=%.3f prepare_sec=%.3f",
                request_id,
                feature_lock_wait_sec,
                time.perf_counter() - prepare_started_at,
            )

        ar_outputs = None
        if params.convert_style:
            ar_started_at = time.perf_counter()
            ar_outputs = []
            for prompt_text, prompt_target in self._build_ar_chunks(source_features, timbre_features, params):
                ar_request = ARGenerateRequest(
                    request_id=request_id,
                    prompt_text=prompt_text,
                    prompt_target=prompt_target,
                    params=params,
                )
                ar_outputs.append(await self.ar_scheduler.submit(ar_request))
            logger.info(
                "request=%s stage=ar_done ar_sec=%.3f chunks=%s",
                request_id,
                time.perf_counter() - ar_started_at,
                len(ar_outputs),
            )

        audio = await self.cfm_scheduler.submit(
            CFMJob(
                request_id=request_id,
                params=params,
                source_features=source_features,
                timbre_features=timbre_features,
                ar_outputs=ar_outputs,
            )
        )
        sample_rate, waveform = audio
        logger.info(
            "request=%s stage=convert_done total_sec=%.3f output_samples=%s",
            request_id,
            time.perf_counter() - total_started_at,
            len(waveform),
        )
        return request_id, sample_rate, waveform

    def _prepare_features(
        self,
        request_id: str,
        source_audio_path: str,
        target_audio_path: str,
        require_source_narrow: bool,
    ) -> Tuple[SourceFeatures, TimbreFeatures]:
        timbre_started_at = time.perf_counter()
        timbre = self._get_or_compute_timbre_features(request_id, target_audio_path)
        logger.info(
            "request=%s stage=timbre_done elapsed_sec=%.3f cache_key=%s",
            request_id,
            time.perf_counter() - timbre_started_at,
            timbre.cache_key,
        )
        source_started_at = time.perf_counter()
        source = self._compute_source_features(source_audio_path, require_source_narrow)
        logger.info(
            "request=%s stage=source_features_done elapsed_sec=%.3f source_mel_len=%s require_narrow=%s",
            request_id,
            time.perf_counter() - source_started_at,
            source.source_mel_len,
            require_source_narrow,
        )
        return source, timbre

    def _get_or_compute_timbre_features(self, request_id: str, target_audio_path: str) -> TimbreFeatures:
        md5_started_at = time.perf_counter()
        cache_key = self._target_audio_cache_key(target_audio_path)
        logger.info(
            "request=%s stage=timbre_md5_done elapsed_sec=%.3f",
            request_id,
            time.perf_counter() - md5_started_at,
        )
        cached = self.timbre_cache.get(cache_key)
        if cached is not None:
            logger.info("request=%s stage=timbre_cache_hit cache_key=%s", request_id, cache_key)
            return cached
        logger.info("request=%s stage=timbre_cache_miss cache_key=%s", request_id, cache_key)
        compute_started_at = time.perf_counter()
        features = self._compute_timbre_features(cache_key, target_audio_path)
        self.timbre_cache.put(cache_key, features)
        logger.info(
            "request=%s stage=timbre_compute_done elapsed_sec=%.3f",
            request_id,
            time.perf_counter() - compute_started_at,
        )
        return features

    @staticmethod
    def _target_audio_cache_key(target_audio_path: str) -> str:
        digest = hashlib.md5()
        with Path(target_audio_path).expanduser().open("rb") as audio_file:
            for chunk in iter(lambda: audio_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"md5:{digest.hexdigest()}"

    @torch.no_grad()
    @torch.inference_mode()
    def _compute_timbre_features(self, cache_key: str, target_audio_path: str) -> TimbreFeatures:
        target_wave = librosa.load(target_audio_path, sr=self.vc_wrapper.sr)[0]
        target_wave = target_wave[: self.vc_wrapper.sr * (self.vc_wrapper.dit_max_context_len - 5)]
        target_wave_tensor = torch.tensor(target_wave).unsqueeze(0).float().to(self.device)
        target_wave_16k = librosa.resample(target_wave, orig_sr=self.vc_wrapper.sr, target_sr=16000)
        target_wave_16k_tensor = torch.tensor(target_wave_16k).unsqueeze(0).to(self.device)

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            target_mel = self.vc_wrapper.mel_fn(target_wave_tensor)
            target_mel_len = target_mel.size(2)
            target_content_indices = self.vc_wrapper._process_content_features(target_wave_16k_tensor, is_narrow=False)
            target_narrow_indices = self.vc_wrapper._process_content_features(target_wave_16k_tensor, is_narrow=True)
            target_narrow_reduced, _ = self.vc_wrapper.duration_reduction_func(target_narrow_indices[0], 1)
            target_style = self.vc_wrapper.compute_style(target_wave_16k_tensor)
            prompt_condition, _ = self.vc_wrapper.cfm_length_regulator(
                target_content_indices,
                ylens=torch.LongTensor([target_mel_len]).to(self.device),
            )

        return TimbreFeatures(
            cache_key=cache_key,
            target_audio_path=target_audio_path,
            target_mel=target_mel,
            target_mel_len=target_mel_len,
            target_content_indices=target_content_indices,
            target_narrow_reduced=target_narrow_reduced,
            target_style=target_style,
            prompt_condition=prompt_condition,
        )

    @torch.no_grad()
    @torch.inference_mode()
    def _compute_source_features(self, source_audio_path: str, require_narrow: bool) -> SourceFeatures:
        source_wave = librosa.load(source_audio_path, sr=self.vc_wrapper.sr)[0]
        source_wave_tensor = torch.tensor(source_wave).unsqueeze(0).float().to(self.device)
        source_wave_16k = librosa.resample(source_wave, orig_sr=self.vc_wrapper.sr, target_sr=16000)
        source_wave_16k_tensor = torch.tensor(source_wave_16k).unsqueeze(0).to(self.device)

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            source_mel = self.vc_wrapper.mel_fn(source_wave_tensor)
            source_content_indices = self.vc_wrapper._process_content_features(source_wave_16k_tensor, is_narrow=False)
            source_narrow_reduced = None
            if require_narrow:
                source_narrow_indices = self.vc_wrapper._process_content_features(source_wave_16k_tensor, is_narrow=True)
                source_narrow_reduced, _ = self.vc_wrapper.duration_reduction_func(source_narrow_indices[0], 1)

        return SourceFeatures(
            source_audio_path=source_audio_path,
            source_mel_len=source_mel.size(2),
            source_content_indices=source_content_indices,
            source_narrow_reduced=source_narrow_reduced,
        )

    def _build_ar_chunks(
        self,
        source: SourceFeatures,
        timbre: TimbreFeatures,
        params: ConcurrentInferenceParams,
    ) -> Sequence[Tuple[torch.Tensor, torch.Tensor]]:
        if source.source_narrow_reduced is None:
            raise RuntimeError("source narrow features are required for convert_style=True")

        if params.anonymization_only:
            prefix_len = 0
            prompt_target = torch.zeros([1, 0], dtype=torch.long, device=self.device)
        else:
            prefix_len = int(timbre.target_narrow_reduced.numel())
            prompt_target = timbre.target_content_indices

        max_chunk_size = max(1, self.vc_wrapper.ar_max_content_len - prefix_len)
        chunks = []
        for start in range(0, int(source.source_narrow_reduced.numel()), max_chunk_size):
            chunk = source.source_narrow_reduced[start:start + max_chunk_size]
            if params.anonymization_only:
                ar_tokens = chunk
            else:
                ar_tokens = torch.cat([timbre.target_narrow_reduced, chunk], dim=0)
            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    prompt_text = self.vc_wrapper.ar_length_regulator(ar_tokens[None])[0]
            chunks.append((prompt_text, prompt_target))
        return chunks

    def metrics(self) -> Dict[str, Dict[str, float]]:
        return {
            "ar_scheduler": self.ar_scheduler.metrics(),
            "cfm_scheduler": self.cfm_scheduler.metrics(),
            "timbre_cache": self.timbre_cache.metrics(),
        }
