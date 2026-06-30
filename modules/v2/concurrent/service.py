import asyncio
import hashlib
import logging
import time
import uuid
from collections import OrderedDict, deque
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import torch


logger = logging.getLogger(__name__)

AR_DECODE_SAFETY_TOKENS = 64
CFM_CHUNK_SAFETY_FRAMES = 16
CFM_BATCH_MAX_SIZE = 4
CFM_BATCH_WAIT_SEC = 0.32


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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
    chunk_index: int
    prompt_text: torch.Tensor
    prompt_target: torch.Tensor
    params: ConcurrentInferenceParams
    future: Optional[asyncio.Future] = None
    slot_id: Optional[int] = None
    is_prefilled: bool = False
    next_input_pos: Optional[torch.Tensor] = None
    next_kv_pos: Optional[torch.Tensor] = None
    next_input_pos_value: int = 0
    next_kv_pos_value: int = 0
    last_emb: Optional[torch.Tensor] = None
    generated_tokens: List[torch.Tensor] = field(default_factory=list)
    generated_token_buffer: Optional[torch.Tensor] = None
    generated_token_count: int = 0
    previous_token_mask: Optional[torch.Tensor] = None
    queued_at: float = field(default_factory=time.perf_counter)
    activated_at: float = 0.0
    prefill_build_sec: float = 0.0
    prefill_decode_sec: float = 0.0
    prefill_embed_sec: float = 0.0
    decode_prepare_sec: float = 0.0
    decode_step_sec: float = 0.0
    decode_embed_sec: float = 0.0
    decode_steps: int = 0
    compiled_decode_steps: int = 0
    eager_decode_steps: int = 0
    last_decode_route: str = ""


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

    def clear(self) -> None:
        self._items.clear()
        self.hits = 0
        self.misses = 0

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
        device: torch.device,
        dtype: torch.dtype,
        max_slots: int,
        max_seq_len: int = 4096,
        min_tokens_before_eos: int = 10,
        max_new_tokens: int = 4000,
        enable_profiling: bool = False,
        compile_decode: bool = False,
        compile_decode_cudagraphs: bool = False,
        compile_batch_sizes: Sequence[int] = (1, 2, 4),
    ):
        self.ar_wrapper = ar_wrapper
        self.device = device
        self.dtype = dtype
        self.max_slots = max_slots
        self.max_seq_len = max_seq_len
        self.min_tokens_before_eos = min_tokens_before_eos
        self.max_new_tokens = max_new_tokens
        self.enable_profiling = enable_profiling
        self.compile_decode_requested = bool(compile_decode)
        self.compile_decode_enabled = bool(compile_decode and device.type == "cuda" and hasattr(torch, "compile"))
        self.compile_decode_cudagraphs = bool(compile_decode_cudagraphs)
        if self.compile_decode_cudagraphs and self._is_t4_device():
            logger.warning(
                "stage=ar_compile_cudagraphs_disabled reason=t4_known_slow device=%s",
                self._device_name(),
            )
            self.compile_decode_cudagraphs = False
        self.compile_batch_sizes = tuple(sorted({
            int(batch_size) for batch_size in compile_batch_sizes
            if 1 <= int(batch_size) <= max_slots
        }))
        self.compiled_decode_fns: Dict[int, Callable] = {}
        self.compile_failures = 0
        self.compile_fallbacks = 0
        self.waiting_queue: "asyncio.Queue[ARGenerateRequest]" = asyncio.Queue()
        self.active: List[ARGenerateRequest] = []
        self.free_slots = list(range(max_slots))
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self.completed = 0
        self.failed = 0
        if self.compile_decode_requested and not self.compile_decode_enabled:
            logger.warning(
                "stage=ar_compile_disabled reason=unsupported_device device=%s",
                device,
            )
        if self.compile_decode_enabled:
            self._build_compiled_decode_fns()
            logger.info(
                "stage=ar_compile_enabled batches=%s max_slots=%s",
                ",".join(str(batch_size) for batch_size in self.compiled_decode_fns),
                self.max_slots,
            )
        logger.info(
            (
                "stage=ar_scheduler_config max_slots=%s max_seq_len=%s device=%s dtype=%s "
                "compile_requested=%s compile_enabled=%s compile_cudagraphs=%s compile_batches=%s profiling=%s"
            ),
            self.max_slots,
            self.max_seq_len,
            self.device,
            self.dtype,
            self.compile_decode_requested,
            self.compile_decode_enabled,
            self.compile_decode_cudagraphs,
            ",".join(str(batch_size) for batch_size in self.compile_batch_sizes) or "none",
            self.enable_profiling,
        )

    def _device_name(self) -> str:
        if self.device.type != "cuda":
            return str(self.device)
        try:
            return torch.cuda.get_device_name(self.device)
        except Exception:
            return str(self.device)

    def _is_t4_device(self) -> bool:
        if self.device.type != "cuda":
            return False
        try:
            major, minor = torch.cuda.get_device_capability(self.device)
            device_name = torch.cuda.get_device_name(self.device).lower()
        except Exception:
            return False
        return (major, minor) == (7, 5) and "t4" in device_name

    async def start(self) -> None:
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._schedule_loop())
        logger.info(
            "stage=ar_scheduler_started max_slots=%s compile_enabled=%s compiled_batches=%s",
            self.max_slots,
            self.compile_decode_enabled,
            ",".join(str(batch_size) for batch_size in sorted(self.compiled_decode_fns)) or "none",
        )

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def _build_compiled_decode_fns(self) -> None:
        self.compiled_decode_fns.clear()
        compile_kwargs = {
            "fullgraph": True,
            "backend": "inductor",
        }
        if self.compile_decode_cudagraphs:
            compile_kwargs["mode"] = "reduce-overhead"
        else:
            compile_kwargs["options"] = {"triton.cudagraphs": False}
        logger.info(
            "stage=ar_compile_config backend=%s mode=%s triton_cudagraphs=%s cudagraphs_requested=%s",
            compile_kwargs["backend"],
            compile_kwargs.get("mode"),
            compile_kwargs.get("options", {}).get("triton.cudagraphs", "default"),
            self.compile_decode_cudagraphs,
        )
        for batch_size in self.compile_batch_sizes:
            batch_size = int(batch_size)

            def decode_forward(x, input_pos, kv_pos, slot_ids):
                return self.ar_wrapper.model.forward_generate(
                    x,
                    input_pos,
                    kv_pos,
                    slot_ids=slot_ids,
                )

            self.compiled_decode_fns[batch_size] = torch.compile(
                decode_forward,
                **compile_kwargs,
            )

    @torch.no_grad()
    def warmup_compiled_decode(self) -> List[int]:
        if not self.compile_decode_enabled:
            return []

        warmed_batches = []
        hidden_size = int(self.ar_wrapper.model.config.dim)
        input_dtype = self.ar_wrapper.sep_token_emb.dtype
        for batch_size, compiled_fn in list(self.compiled_decode_fns.items()):
            started_at = time.perf_counter()
            x = torch.zeros(
                (batch_size, 1, hidden_size),
                device=self.device,
                dtype=input_dtype,
            )
            input_pos = torch.zeros((batch_size, 1), device=self.device, dtype=torch.long)
            kv_pos = torch.zeros((batch_size, 1), device=self.device, dtype=torch.long)
            slot_ids = torch.arange(batch_size, device=self.device, dtype=torch.long)
            try:
                logger.info("stage=warmup_ar_decode_start batch_size=%s", batch_size)
                with self._autocast_context():
                    result = compiled_fn(x, input_pos, kv_pos, slot_ids)
                synchronize_device(self.device)
                del result
                warmed_batches.append(batch_size)
                logger.info(
                    "stage=warmup_ar_decode_done batch_size=%s elapsed_sec=%.3f",
                    batch_size,
                    time.perf_counter() - started_at,
                )
            except Exception as exc:
                self.compile_failures += 1
                self.compiled_decode_fns.pop(batch_size, None)
                logger.warning(
                    "stage=warmup_ar_decode_failed batch_size=%s error_type=%s error=%s",
                    batch_size,
                    type(exc).__name__,
                    exc,
                )

        if not self.compiled_decode_fns:
            self.compile_decode_enabled = False
            logger.warning("stage=ar_compile_disabled reason=all_warmups_failed")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info(
            (
                "stage=warmup_ar_decode_summary requested_batches=%s warmed_batches=%s "
                "enabled=%s remaining_batches=%s failures=%s"
            ),
            ",".join(str(batch_size) for batch_size in self.compile_batch_sizes) or "none",
            ",".join(str(batch_size) for batch_size in warmed_batches) or "none",
            self.compile_decode_enabled,
            ",".join(str(batch_size) for batch_size in sorted(self.compiled_decode_fns)) or "none",
            self.compile_failures,
        )
        return warmed_batches

    async def submit(self, request: ARGenerateRequest) -> torch.Tensor:
        request.future = asyncio.get_running_loop().create_future()
        request.queued_at = time.perf_counter()
        logger.info(
            (
                "request=%s stage=ar_queue_submit chunk=%s queue_size=%s active=%s free_slots=%s "
                "prompt_text_len=%s prompt_target_len=%s"
            ),
            request.request_id,
            request.chunk_index,
            self.waiting_queue.qsize(),
            len(self.active),
            len(self.free_slots),
            request.prompt_text.size(1),
            request.prompt_target.size(-1),
        )
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
        request.activated_at = time.perf_counter()
        self.active.append(request)
        logger.info(
            (
                "request=%s stage=ar_request_activated chunk=%s slot=%s queue_wait_sec=%.3f "
                "active=%s queue_size=%s free_slots=%s"
            ),
            request.request_id,
            request.chunk_index,
            request.slot_id,
            request.activated_at - request.queued_at,
            len(self.active),
            self.waiting_queue.qsize(),
            len(self.free_slots),
        )

    @torch.no_grad()
    def _prefill_one(self, request: ARGenerateRequest) -> None:
        assert request.slot_id is not None
        profile_started_at = self._profile_start()
        with self._autocast_context():
            emb_seq, input_pos, kv_pos = self.ar_wrapper.build_generation_inputs(
                request.prompt_text,
                request.prompt_target,
            )
        request.prefill_build_sec += self._profile_elapsed(profile_started_at)
        if emb_seq.size(1) >= self.max_seq_len:
            raise RuntimeError(
                f"AR prompt is too long: {emb_seq.size(1)} >= max_seq_len {self.max_seq_len}"
            )

        eos_token = self.ar_wrapper.model.config.vocab_size - 1
        profile_started_at = self._profile_start()
        with self._autocast_context():
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
        request.prefill_decode_sec += self._profile_elapsed(profile_started_at)
        token = next_tokens[0].reshape(()).clone()
        request.generated_token_buffer = torch.empty(
            (self.max_new_tokens,),
            device=token.device,
            dtype=torch.long,
        )
        request.generated_token_buffer[0] = token.long()
        request.generated_token_count = 1
        request.previous_token_mask = torch.zeros(
            (int(self.ar_wrapper.model.config.vocab_size),),
            device=token.device,
            dtype=torch.bool,
        )
        request.previous_token_mask[int(token.detach().cpu())] = True
        profile_started_at = self._profile_start()
        with self._autocast_context():
            request.last_emb = self.ar_wrapper.embed_generated_token(token)
        request.prefill_embed_sec += self._profile_elapsed(profile_started_at)
        request.next_input_pos_value = int(input_pos[-1].detach().cpu()) + 1
        request.next_kv_pos_value = int(kv_pos[-1].detach().cpu()) + 1
        request.is_prefilled = True
        if self.enable_profiling:
            logger.info(
                (
                    "request=%s stage=ar_prefill_done chunk=%s slot=%s prompt_text_len=%s "
                    "prompt_target_len=%s kv_len=%s build_sec=%.3f decode_sec=%.3f embed_sec=%.3f"
                ),
                request.request_id,
                request.chunk_index,
                request.slot_id,
                request.prompt_text.size(1),
                request.prompt_target.size(-1),
                emb_seq.size(1),
                request.prefill_build_sec,
                request.prefill_decode_sec,
                request.prefill_embed_sec,
            )

    @torch.no_grad()
    def _decode_one_step(self, requests: Sequence[ARGenerateRequest]) -> None:
        eos_token = self.ar_wrapper.model.config.vocab_size - 1
        for request in list(requests):
            if request.generated_token_count >= self.max_new_tokens:
                self._finish(request, reason="max_new_tokens")

        live_requests = [
            request for request in requests
            if request in self.active and request.last_emb is not None and request.generated_token_count < self.max_new_tokens
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
            profile_started_at = self._profile_start()
            x = torch.cat([request.last_emb for request in group], dim=0)
            input_pos = torch.tensor(
                [[request.next_input_pos_value] for request in group],
                device=self.device,
                dtype=torch.long,
            )
            kv_pos = torch.tensor(
                [[request.next_kv_pos_value] for request in group],
                device=self.device,
                dtype=torch.long,
            )
            if any(request.slot_id is None for request in group):
                raise RuntimeError("AR request is missing slot_id during decode")
            slot_ids_list = [int(request.slot_id) for request in group]
            slot_ids = torch.tensor(slot_ids_list, device=self.device, dtype=torch.long)
            previous_tokens = [
                request.generated_token_buffer[:request.generated_token_count]
                for request in group
            ]
            previous_token_masks = [request.previous_token_mask for request in group]
            suppress_tokens = [
                [eos_token] if request.generated_token_count < self.min_tokens_before_eos else None
                for request in group
            ]
            compiled_decode_fn, decode_bucket = self._select_compiled_decode(len(group))
            decode_mode = "compiled" if compiled_decode_fn is not None else "eager"
            active_count = len(group)
            padded_count = 0
            if compiled_decode_fn is not None and decode_bucket is not None and decode_bucket > len(group):
                padded = self._pad_decode_batch(x, input_pos, kv_pos, slot_ids, decode_bucket, slot_ids_list)
                if padded is None:
                    if self.enable_profiling:
                        logger.info(
                            (
                                "stage=ar_decode_padding_skipped active_count=%s bucket=%s free_slots=%s "
                                "used_slots=%s route=eager"
                            ),
                            active_count,
                            decode_bucket,
                            len(self.free_slots),
                            ",".join(str(slot_id) for slot_id in slot_ids_list),
                        )
                    compiled_decode_fn = None
                    decode_bucket = None
                    decode_mode = "eager"
                else:
                    x, input_pos, kv_pos, slot_ids = padded
                    padded_count = int(x.size(0)) - active_count
            prepare_sec = self._profile_elapsed(profile_started_at)
            for request in group:
                request.decode_prepare_sec += prepare_sec / len(group)
            self._log_decode_route_if_changed(
                group,
                decode_mode=decode_mode,
                active_count=active_count,
                batch_size=int(x.size(0)),
                decode_bucket=decode_bucket,
                padded_count=padded_count,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )

            profile_started_at = self._profile_start()
            with self._autocast_context():
                try:
                    next_tokens = self.ar_wrapper.decode_one_token_ar_batch(
                        x,
                        input_pos,
                        kv_pos,
                        slot_ids=slot_ids,
                        previous_tokens=previous_tokens,
                        previous_token_masks=previous_token_masks,
                        suppress_tokens=suppress_tokens,
                        compiled_decode_fn=compiled_decode_fn,
                        active_count=active_count,
                        top_p=top_p,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                    )
                except Exception as exc:
                    if compiled_decode_fn is None or decode_bucket is None:
                        raise
                    self.compile_fallbacks += 1
                    self.compiled_decode_fns.pop(decode_bucket, None)
                    if not self.compiled_decode_fns:
                        self.compile_decode_enabled = False
                    logger.warning(
                        (
                            "stage=ar_decode_compile_fallback batch_size=%s active_count=%s "
                            "requests=%s chunks=%s error_type=%s error=%s"
                        ),
                        decode_bucket,
                        active_count,
                        ",".join(request.request_id for request in group),
                        ",".join(str(request.chunk_index) for request in group),
                        type(exc).__name__,
                        exc,
                    )
                    next_tokens = self.ar_wrapper.decode_one_token_ar_batch(
                        torch.cat([request.last_emb for request in group], dim=0),
                        torch.tensor(
                            [[request.next_input_pos_value] for request in group],
                            device=self.device,
                            dtype=torch.long,
                        ),
                        torch.tensor(
                            [[request.next_kv_pos_value] for request in group],
                            device=self.device,
                            dtype=torch.long,
                        ),
                        slot_ids=torch.tensor(slot_ids_list, device=self.device, dtype=torch.long),
                        previous_tokens=previous_tokens,
                        previous_token_masks=previous_token_masks,
                        suppress_tokens=suppress_tokens,
                        top_p=top_p,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                    )
                    decode_mode = "eager_fallback"
            step_sec = self._profile_elapsed(profile_started_at)
            for request in group:
                request.decode_step_sec += step_sec / len(group)
                request.decode_steps += 1
                if decode_mode == "compiled":
                    request.compiled_decode_steps += 1
                else:
                    request.eager_decode_steps += 1

            next_tokens = next_tokens[:len(group)]
            next_token_values = next_tokens.detach().cpu().tolist()
            embed_requests: List[ARGenerateRequest] = []
            embed_tokens: List[torch.Tensor] = []
            for request, token, token_value in zip(group, next_tokens, next_token_values):
                token = token.reshape(()).clone()
                reached_eos = int(token_value) == eos_token and request.generated_token_count >= self.min_tokens_before_eos
                reached_limit = request.generated_token_count + 1 >= self.max_new_tokens
                reached_cache_limit = request.next_kv_pos_value + 1 >= self.max_seq_len
                if reached_eos or reached_limit or reached_cache_limit:
                    reason = "eos" if reached_eos else "max_new_tokens" if reached_limit else "cache_limit"
                    self._finish(request, reason=reason)
                    continue

                if request.generated_token_buffer is None:
                    raise RuntimeError("AR request is missing generated token buffer")
                request.generated_token_buffer[request.generated_token_count] = token.long()
                request.generated_token_count += 1
                if request.previous_token_mask is not None:
                    request.previous_token_mask[int(token_value)] = True
                request.next_input_pos_value += 1
                request.next_kv_pos_value += 1
                embed_requests.append(request)
                embed_tokens.append(token)

            if embed_requests:
                profile_started_at = self._profile_start()
                with self._autocast_context():
                    embedded_tokens = self.ar_wrapper.embed_generated_tokens(torch.stack(embed_tokens, dim=0))
                embed_sec = self._profile_elapsed(profile_started_at)
                for embed_index, request in enumerate(embed_requests):
                    request.last_emb = embedded_tokens[embed_index:embed_index + 1]
                    request.decode_embed_sec += embed_sec / len(embed_requests)

    def _log_decode_route_if_changed(
        self,
        requests: Sequence[ARGenerateRequest],
        *,
        decode_mode: str,
        active_count: int,
        batch_size: int,
        decode_bucket: Optional[int],
        padded_count: int,
        top_p: float,
        temperature: float,
        repetition_penalty: float,
    ) -> None:
        if not self.enable_profiling:
            return
        route_key = f"{decode_mode}:{active_count}:{batch_size}:{decode_bucket}:{padded_count}"
        changed_requests = [request for request in requests if request.last_decode_route != route_key]
        if not changed_requests:
            return
        for request in changed_requests:
            request.last_decode_route = route_key
        logger.info(
            (
                "stage=ar_decode_route mode=%s active_count=%s batch_size=%s bucket=%s padded=%s "
                "free_slots=%s requests=%s chunks=%s slots=%s top_p=%.3f temperature=%.3f repetition_penalty=%.3f"
            ),
            decode_mode,
            active_count,
            batch_size,
            decode_bucket if decode_bucket is not None else "none",
            padded_count,
            len(self.free_slots),
            ",".join(request.request_id for request in requests),
            ",".join(str(request.chunk_index) for request in requests),
            ",".join(str(request.slot_id) for request in requests),
            top_p,
            temperature,
            repetition_penalty,
        )

    def _autocast_context(self):
        if self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16):
            return torch.autocast(device_type=self.device.type, dtype=self.dtype)
        return nullcontext()

    def _select_compiled_decode(self, active_count: int) -> Tuple[Optional[Callable], Optional[int]]:
        if not self.compile_decode_enabled:
            return None, None
        for batch_size in sorted(self.compiled_decode_fns):
            if active_count <= batch_size:
                return self.compiled_decode_fns[batch_size], batch_size
        return None, None

    def _pad_decode_batch(
        self,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        kv_pos: torch.Tensor,
        slot_ids: torch.Tensor,
        batch_size: int,
        used_slot_ids: Sequence[int],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        pad_count = batch_size - x.size(0)
        if pad_count <= 0:
            return x, input_pos, kv_pos, slot_ids

        used_slots = set(int(slot_id) for slot_id in used_slot_ids)
        dummy_slots = [slot_id for slot_id in self.free_slots if slot_id not in used_slots]
        if len(dummy_slots) < pad_count:
            return None

        pad_x = torch.zeros(
            (pad_count, *x.shape[1:]),
            device=x.device,
            dtype=x.dtype,
        )
        pad_input_pos = input_pos[-1:].expand(pad_count, -1).clone()
        pad_kv_pos = kv_pos[-1:].expand(pad_count, -1).clone()
        pad_slot_ids = torch.tensor(dummy_slots[:pad_count], device=self.device, dtype=torch.long)
        return (
            torch.cat([x, pad_x], dim=0),
            torch.cat([input_pos, pad_input_pos], dim=0),
            torch.cat([kv_pos, pad_kv_pos], dim=0),
            torch.cat([slot_ids, pad_slot_ids], dim=0),
        )

    def _profile_start(self) -> float:
        if not self.enable_profiling:
            return 0.0
        synchronize_device(self.device)
        return time.perf_counter()

    def _profile_elapsed(self, started_at: float) -> float:
        if not self.enable_profiling:
            return 0.0
        synchronize_device(self.device)
        return time.perf_counter() - started_at

    def _finish(self, request: ARGenerateRequest, reason: str = "completed") -> None:
        if request in self.active:
            self.active.remove(request)
        finished_at = time.perf_counter()
        if request.slot_id is not None:
            self.free_slots.append(request.slot_id)
            self.free_slots.sort()
        if request.generated_token_buffer is not None and request.generated_token_count > 0:
            result = request.generated_token_buffer[:request.generated_token_count].long().unsqueeze(0)
        elif request.generated_tokens:
            result = torch.stack([token.reshape(()) for token in request.generated_tokens], dim=0).long().unsqueeze(0)
        else:
            result = torch.empty((1, 0), device=self.device, dtype=torch.long)
        if request.future is not None and not request.future.done():
            request.future.set_result(result)
        self.completed += 1
        if self.enable_profiling:
            total_sec = finished_at - request.activated_at if request.activated_at else finished_at - request.queued_at
            queue_wait_sec = request.activated_at - request.queued_at if request.activated_at else 0.0
            generated_tokens = request.generated_token_count
            tokens_per_sec = generated_tokens / total_sec if total_sec > 0 else 0.0
            logger.info(
                (
                    "request=%s stage=ar_chunk_done chunk=%s reason=%s slot=%s total_sec=%.3f "
                    "queue_wait_sec=%.3f prompt_text_len=%s prompt_target_len=%s generated_tokens=%s "
                    "decode_steps=%s compiled_decode_steps=%s eager_decode_steps=%s tokens_per_sec=%.2f prefill_build_sec=%.3f "
                    "prefill_decode_sec=%.3f prefill_embed_sec=%.3f decode_prepare_sec=%.3f "
                    "decode_step_sec=%.3f decode_embed_sec=%.3f"
                ),
                request.request_id,
                request.chunk_index,
                reason,
                request.slot_id,
                total_sec,
                queue_wait_sec,
                request.prompt_text.size(1),
                request.prompt_target.size(-1),
                generated_tokens,
                request.decode_steps,
                request.compiled_decode_steps,
                request.eager_decode_steps,
                tokens_per_sec,
                request.prefill_build_sec,
                request.prefill_decode_sec,
                request.prefill_embed_sec,
                request.decode_prepare_sec,
                request.decode_step_sec,
                request.decode_embed_sec,
            )

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
        logger.exception(
            (
                "request=%s stage=ar_request_failed chunk=%s slot=%s generated_tokens=%s "
                "active=%s queue_size=%s free_slots=%s error_type=%s error=%s"
            ),
            request.request_id,
            request.chunk_index,
            request.slot_id,
            request.generated_token_count,
            len(self.active),
            self.waiting_queue.qsize(),
            len(self.free_slots),
            type(exc).__name__,
            exc,
        )

    def metrics(self) -> Dict[str, int]:
        return {
            "max_slots": self.max_slots,
            "active_requests": len(self.active),
            "queue_length": self.waiting_queue.qsize(),
            "free_slots": len(self.free_slots),
            "completed": self.completed,
            "failed": self.failed,
            "compile_decode_enabled": int(self.compile_decode_enabled),
            "compiled_decode_batches": list(sorted(self.compiled_decode_fns)),
            "compile_failures": self.compile_failures,
            "compile_fallbacks": self.compile_fallbacks,
        }


class CFMScheduler:
    def __init__(
        self,
        vc_wrapper,
        device: torch.device,
        dtype: torch.dtype,
        max_concurrent: int = 1,
        enable_profiling: bool = False,
    ):
        self.vc_wrapper = vc_wrapper
        self.device = device
        self.dtype = dtype
        self.max_concurrent = max(1, max_concurrent)
        self.enable_profiling = enable_profiling
        self.queue: "asyncio.Queue[CFMJob]" = asyncio.Queue()
        self._deferred_jobs: Deque[CFMJob] = deque()
        self._tasks: List[asyncio.Task] = []
        self._running = False
        self.active = 0
        self.completed = 0
        self.failed = 0
        self._executor: Optional[ThreadPoolExecutor] = None

    async def start(self) -> None:
        if self._tasks:
            return
        self._running = True
        executor_workers = 1 if self.vc_wrapper.dit_compiled else self.max_concurrent
        self._executor = ThreadPoolExecutor(
            max_workers=executor_workers,
            thread_name_prefix="seed-vc-cfm",
        )
        logger.info(
            "stage=cfm_executor_started workers=%s dit_compiled=%s",
            executor_workers,
            self.vc_wrapper.dit_compiled,
        )
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
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

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
            job = await self._get_next_job()
            jobs = await self._collect_batch_jobs(job)
            self.active += len(jobs)
            run_started_at = time.perf_counter()
            for batch_index, batch_job in enumerate(jobs):
                queue_wait_sec = run_started_at - batch_job.submitted_at
                logger.info(
                    "request=%s stage=cfm_start queue_wait_sec=%.3f active=%s queue_size=%s batch_size=%s batch_index=%s",
                    batch_job.request_id,
                    queue_wait_sec,
                    self.active,
                    self.queue.qsize() + len(self._deferred_jobs),
                    len(jobs),
                    batch_index,
                )
            try:
                results = await self._run_in_executor(self._run_jobs, jobs)
                run_sec = time.perf_counter() - run_started_at
                for batch_job, result in zip(jobs, results):
                    if batch_job.future is not None and not batch_job.future.done():
                        batch_job.future.set_result(result)
                    self.completed += 1
                    logger.info(
                        "request=%s stage=cfm_done run_sec=%.3f completed=%s batch_size=%s",
                        batch_job.request_id,
                        run_sec,
                        self.completed,
                        len(jobs),
                    )
            except Exception as exc:
                for batch_job in jobs:
                    if batch_job.future is not None and not batch_job.future.done():
                        batch_job.future.set_exception(exc)
                    self.failed += 1
                    logger.exception(
                        "request=%s stage=cfm_failed run_sec=%.3f batch_size=%s",
                        batch_job.request_id,
                        time.perf_counter() - run_started_at,
                        len(jobs),
                    )
            finally:
                self.active -= len(jobs)

    async def _get_next_job(self) -> CFMJob:
        if self._deferred_jobs:
            return self._deferred_jobs.popleft()
        return await self.queue.get()

    async def _collect_batch_jobs(self, first_job: CFMJob) -> List[CFMJob]:
        if not self._is_batchable_style_job(first_job):
            return [first_job]

        jobs = [first_job]
        deadline = time.perf_counter() + CFM_BATCH_WAIT_SEC
        while len(jobs) < CFM_BATCH_MAX_SIZE:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                next_job = await asyncio.wait_for(self.queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            if self._jobs_batch_compatible(first_job, next_job):
                jobs.append(next_job)
                continue
            self._deferred_jobs.append(next_job)
            break
        return jobs

    async def _run_in_executor(self, func, *args):
        if self._executor is None:
            raise RuntimeError("CFM scheduler is not started")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, func, *args)

    @torch.no_grad()
    @torch.inference_mode()
    def _run_job(self, job: CFMJob) -> Tuple[int, np.ndarray]:
        if job.params.convert_style:
            return self._run_style_job(job)
        return self._run_timbre_job(job)

    @torch.no_grad()
    @torch.inference_mode()
    def _run_jobs(self, jobs: Sequence[CFMJob]) -> List[Tuple[int, np.ndarray]]:
        if len(jobs) > 1 and all(self._jobs_batch_compatible(jobs[0], job) for job in jobs):
            return self._run_style_batch_jobs(jobs)
        return [self._run_job(job) for job in jobs]

    def _is_batchable_style_job(self, job: CFMJob) -> bool:
        if not job.params.convert_style or not job.ar_outputs or len(job.ar_outputs) != 1:
            return False
        source = job.source_features
        timbre = job.timbre_features
        if source.source_content_indices.size(0) != 1:
            return False
        max_condition_len = int(self.vc_wrapper.cfm_compile_buckets[-1])
        max_source_window = max(1, max_condition_len - int(timbre.target_mel_len))
        ar_out = job.ar_outputs[0]
        ar_out_mel_len = int(
            source.source_mel_len
            / max(1, source.source_content_indices.size(-1))
            * ar_out.size(-1)
            * job.params.length_adjust
        )
        return 0 < ar_out_mel_len <= max_source_window

    def _jobs_batch_compatible(self, first: CFMJob, other: CFMJob) -> bool:
        if not self._is_batchable_style_job(other):
            return False
        if not self._is_batchable_style_job(first):
            return False
        first_params = first.params
        other_params = other.params
        if (
            first_params.diffusion_steps != other_params.diffusion_steps
            or first_params.length_adjust != other_params.length_adjust
            or first_params.intelligibility_cfg_rate != other_params.intelligibility_cfg_rate
            or first_params.similarity_cfg_rate != other_params.similarity_cfg_rate
            or first_params.anonymization_only != other_params.anonymization_only
        ):
            return False
        first_timbre = first.timbre_features
        other_timbre = other.timbre_features
        return (
            first_timbre.cache_key == other_timbre.cache_key
            and int(first_timbre.target_mel_len) == int(other_timbre.target_mel_len)
            and first_timbre.target_mel.shape == other_timbre.target_mel.shape
            and first_timbre.prompt_condition.shape == other_timbre.prompt_condition.shape
            and first_timbre.target_style.shape == other_timbre.target_style.shape
        )

    def _run_timbre_job(self, job: CFMJob) -> Tuple[int, np.ndarray]:
        source = job.source_features
        regulator_started_at = time.perf_counter()
        self._synchronize_for_profiling()
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            cond, _ = self.vc_wrapper.cfm_length_regulator(
                source.source_content_indices,
                ylens=torch.LongTensor([source.source_mel_len]).to(self.device),
            )
        self._synchronize_for_profiling()
        if self.enable_profiling:
            logger.info(
                "request=%s stage=cfm_length_regulator_done elapsed_sec=%.3f cond_len=%s source_mel_len=%s",
                job.request_id,
                time.perf_counter() - regulator_started_at,
                cond.size(1),
                source.source_mel_len,
            )
        return self._render_condition_chunks(job, cond)

    def _run_style_batch_jobs(self, jobs: Sequence[CFMJob]) -> List[Tuple[int, np.ndarray]]:
        if not jobs:
            return []

        first_timbre = jobs[0].timbre_features
        target_mel_len = int(first_timbre.target_mel_len)
        cat_conditions = []
        original_lens = []
        output_mel_lens = []

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            for job in jobs:
                source = job.source_features
                ar_out = job.ar_outputs[0]
                ar_out_mel_len = torch.LongTensor([
                    int(
                        source.source_mel_len
                        / max(1, source.source_content_indices.size(-1))
                        * ar_out.size(-1)
                        * job.params.length_adjust
                    )
                ]).to(self.device)
                chunk_cond, _ = self.vc_wrapper.cfm_length_regulator(ar_out, ylens=ar_out_mel_len)
                cat_condition = torch.cat([job.timbre_features.prompt_condition, chunk_cond], dim=1)
                original_lens.append(int(cat_condition.size(1)))
                output_mel_lens.append(int(cat_condition.size(1)) - target_mel_len)
                cat_conditions.append(cat_condition)

            max_original_len = max(original_lens)
            compile_bucket_len = max_original_len
            if self.vc_wrapper.dit_compiled:
                compile_bucket_len = self.vc_wrapper.select_cfm_compile_bucket(max_original_len)
            padded_conditions = [
                torch.nn.functional.pad(
                    cat_condition,
                    (0, 0, 0, compile_bucket_len - int(cat_condition.size(1))),
                    value=0,
                )
                for cat_condition in cat_conditions
            ]
            batched_condition = torch.cat(padded_conditions, dim=0)
            x_lens = torch.LongTensor(original_lens).to(self.device)
            target_mel = torch.cat([job.timbre_features.target_mel for job in jobs], dim=0)
            target_style = torch.cat([job.timbre_features.target_style for job in jobs], dim=0)
            vc_mels = self.vc_wrapper.cfm.inference(
                batched_condition,
                x_lens,
                target_mel,
                target_style,
                jobs[0].params.diffusion_steps,
                inference_cfg_rate=[
                    jobs[0].params.intelligibility_cfg_rate,
                    jobs[0].params.similarity_cfg_rate,
                ],
                random_voice=jobs[0].params.anonymization_only,
            )

        results = []
        overlap_wave_len = self.vc_wrapper.overlap_frame_len * self.vc_wrapper.hop_size
        for index, job in enumerate(jobs):
            vc_mel = vc_mels[index:index + 1, :, target_mel_len:original_lens[index]]
            vc_wave = self.vc_wrapper.vocoder(vc_mel).squeeze()[None]
            generated_wave_chunks = []
            processed_frames, previous_chunk, should_break, _, full_audio = self.vc_wrapper._stream_wave_chunks(
                vc_wave,
                0,
                vc_mel,
                overlap_wave_len,
                generated_wave_chunks,
                None,
                True,
                stream_output=False,
            )
            if should_break:
                waveform = full_audio
            else:
                waveform = np.concatenate(generated_wave_chunks)
            results.append((self.vc_wrapper.sr, waveform))

        if self.enable_profiling:
            logger.info(
                (
                    "stage=cfm_batch_done batch_size=%s compile_bucket_len=%s original_lens=%s "
                    "output_mel_lens=%s requests=%s"
                ),
                len(jobs),
                compile_bucket_len,
                ",".join(str(length) for length in original_lens),
                ",".join(str(length) for length in output_mel_lens),
                ",".join(job.request_id for job in jobs),
            )
        return results

    def _run_style_job(self, job: CFMJob) -> Tuple[int, np.ndarray]:
        if not job.ar_outputs:
            raise RuntimeError("convert_style=True requires AR outputs")

        generated_wave_chunks = []
        previous_chunk = None
        processed_frames = 0
        overlap_wave_len = self.vc_wrapper.overlap_frame_len * self.vc_wrapper.hop_size
        max_condition_len = int(self.vc_wrapper.cfm_compile_buckets[-1])

        source = job.source_features
        timbre = job.timbre_features
        max_source_window = max(1, max_condition_len - int(timbre.target_mel_len))
        for index, ar_out in enumerate(job.ar_outputs):
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

            chunk_offset = 0
            while chunk_offset < chunk_cond.size(1):
                cond_slice = chunk_cond[:, chunk_offset:chunk_offset + max_source_window]
                is_last_chunk = (
                    index + 1 >= len(job.ar_outputs)
                    and chunk_offset + max_source_window >= chunk_cond.size(1)
                )
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    cat_condition = torch.cat([timbre.prompt_condition, cond_slice], dim=1)
                    original_len = cat_condition.size(1)
                    vc_mel, _, _ = self._infer_cfm(job, cat_condition, random_voice=job.params.anonymization_only)
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
                chunk_offset += max_source_window
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
            condition_len = cat_condition.size(1)
            cfm_started_at = time.perf_counter()
            self._synchronize_for_profiling()
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                vc_mel, compile_bucket_len, compile_pad_frames = self._infer_cfm(
                    job,
                    cat_condition,
                    random_voice=job.params.anonymization_only,
                )
            self._synchronize_for_profiling()
            cfm_infer_sec = time.perf_counter() - cfm_started_at
            original_len = condition_len
            vc_mel = vc_mel[:, :, timbre.target_mel_len:original_len]
            vocoder_started_at = time.perf_counter()
            self._synchronize_for_profiling()
            vc_wave = self.vc_wrapper.vocoder(vc_mel).squeeze()[None]
            self._synchronize_for_profiling()
            vocoder_sec = time.perf_counter() - vocoder_started_at
            stream_started_at = time.perf_counter()
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
            stream_cpu_sec = time.perf_counter() - stream_started_at
            chunk_sec = time.perf_counter() - chunk_started_at
            if self.enable_profiling:
                logger.info(
                    (
                        "request=%s stage=cfm_chunk_done chunk=%s chunk_sec=%.3f "
                        "cfm_infer_sec=%.3f vocoder_sec=%.3f stream_cpu_sec=%.3f "
                        "condition_len=%s compile_bucket_len=%s compile_pad_frames=%s "
                        "source_chunk_len=%s output_mel_len=%s last=%s"
                    ),
                    job.request_id,
                    chunk_index,
                    chunk_sec,
                    cfm_infer_sec,
                    vocoder_sec,
                    stream_cpu_sec,
                    condition_len,
                    compile_bucket_len,
                    compile_pad_frames,
                    chunk_cond.size(1),
                    vc_mel.size(2),
                    is_last_chunk,
                )
            if should_break:
                return self.vc_wrapper.sr, full_audio
            chunk_index += 1

        return self.vc_wrapper.sr, np.concatenate(generated_wave_chunks)

    def _infer_cfm(self, job: CFMJob, cat_condition: torch.Tensor, random_voice: bool) -> Tuple[torch.Tensor, int, int]:
        original_len = cat_condition.size(1)
        compile_bucket_len = original_len
        compile_pad_frames = 0
        if self.vc_wrapper.dit_compiled:
            compile_bucket_len = self.vc_wrapper.select_cfm_compile_bucket(original_len)
            compile_pad_frames = compile_bucket_len - original_len
            cat_condition = torch.nn.functional.pad(
                cat_condition,
                (0, 0, 0, compile_pad_frames),
                value=0,
            )
        timbre = job.timbre_features
        vc_mel = self.vc_wrapper.cfm.inference(
            cat_condition,
            torch.LongTensor([original_len]).to(self.device),
            timbre.target_mel,
            timbre.target_style,
            job.params.diffusion_steps,
            inference_cfg_rate=[job.params.intelligibility_cfg_rate, job.params.similarity_cfg_rate],
            random_voice=random_voice,
        )
        return vc_mel, compile_bucket_len, compile_pad_frames

    @torch.no_grad()
    @torch.inference_mode()
    def warmup_compile_buckets(self, timbre: TimbreFeatures, params: ConcurrentInferenceParams) -> List[int]:
        if not self.vc_wrapper.dit_compiled:
            logger.info("stage=warmup_cfm_buckets_skipped reason=cfm_not_compiled")
            return []

        warmed_buckets = []
        content_dim = int(timbre.prompt_condition.size(-1))
        buckets = tuple(int(bucket_len) for bucket_len in self.vc_wrapper.cfm_compile_buckets)
        logger.info(
            "stage=warmup_cfm_buckets_start buckets=%s diffusion_steps=%s",
            ",".join(str(bucket_len) for bucket_len in buckets),
            params.diffusion_steps,
        )
        for bucket_len in buckets:
            bucket_started_at = time.perf_counter()
            request_id = f"warmup-cfm-bucket-{bucket_len}"
            prompt_len = min(int(timbre.target_mel.size(-1)), max(1, bucket_len - 1))
            warmup_timbre = TimbreFeatures(
                cache_key=timbre.cache_key,
                target_audio_path=timbre.target_audio_path,
                target_mel=timbre.target_mel[:, :, :prompt_len].contiguous(),
                target_mel_len=prompt_len,
                target_content_indices=timbre.target_content_indices,
                target_narrow_reduced=timbre.target_narrow_reduced,
                target_style=timbre.target_style,
                prompt_condition=timbre.prompt_condition[:, :prompt_len, :].contiguous(),
            )
            dummy_source = SourceFeatures(
                source_audio_path="<cfm_bucket_warmup>",
                source_mel_len=bucket_len,
                source_content_indices=torch.empty(1, 0, dtype=torch.long, device=self.device),
            )
            job = CFMJob(
                request_id=request_id,
                params=params,
                source_features=dummy_source,
                timbre_features=warmup_timbre,
            )
            cat_condition = torch.zeros(
                (1, bucket_len, content_dim),
                dtype=timbre.prompt_condition.dtype,
                device=self.device,
            )
            logger.info(
                "request=%s stage=warmup_cfm_bucket_start bucket_len=%s prompt_len=%s",
                request_id,
                bucket_len,
                prompt_len,
            )
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                vc_mel, compile_bucket_len, compile_pad_frames = self._infer_cfm(
                    job,
                    cat_condition,
                    random_voice=params.anonymization_only,
                )
            synchronize_device(self.device)
            del vc_mel
            warmed_buckets.append(bucket_len)
            logger.info(
                (
                    "request=%s stage=warmup_cfm_bucket_done bucket_len=%s "
                    "compile_bucket_len=%s compile_pad_frames=%s elapsed_sec=%.3f"
                ),
                request_id,
                bucket_len,
                compile_bucket_len,
                compile_pad_frames,
                time.perf_counter() - bucket_started_at,
            )

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return warmed_buckets

    async def warmup_compile_buckets_async(
        self,
        timbre: TimbreFeatures,
        params: ConcurrentInferenceParams,
    ) -> List[int]:
        return await self._run_in_executor(self.warmup_compile_buckets, timbre, params)

    def _synchronize_for_profiling(self) -> None:
        if self.enable_profiling:
            synchronize_device(self.device)

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
        enable_profiling: bool = False,
        compile_ar: bool = False,
        compile_ar_cudagraphs: bool = False,
    ):
        self.vc_wrapper = vc_wrapper
        self.device = device
        self.dtype = dtype
        self.enable_profiling = enable_profiling
        self.ar_scheduler = ARScheduler(
            vc_wrapper.ar,
            device=device,
            dtype=dtype,
            max_slots=ar_slots,
            max_seq_len=ar_max_seq_len,
            enable_profiling=enable_profiling,
            compile_decode=compile_ar,
            compile_decode_cudagraphs=compile_ar_cudagraphs,
        )
        self.cfm_scheduler = CFMScheduler(
            vc_wrapper,
            device=device,
            dtype=dtype,
            max_concurrent=cfm_max_concurrent,
            enable_profiling=enable_profiling,
        )
        self.timbre_cache = TimbreFeatureCache(max_size=timbre_cache_size)
        self._feature_lock = asyncio.Lock()

    async def start(self) -> None:
        await self.ar_scheduler.start()
        await self.cfm_scheduler.start()

    async def stop(self) -> None:
        await self.ar_scheduler.stop()
        await self.cfm_scheduler.stop()

    async def warmup_cfm_compile_buckets(
        self,
        target_audio_path: str,
        params: ConcurrentInferenceParams,
    ) -> List[int]:
        request_id = "warmup-cfm-buckets"
        async with self._feature_lock:
            timbre = await asyncio.to_thread(
                self._get_or_compute_timbre_features,
                request_id,
                target_audio_path,
            )
        return await self.cfm_scheduler.warmup_compile_buckets_async(timbre, params)

    async def warmup_ar_decode(self) -> List[int]:
        return self.ar_scheduler.warmup_compiled_decode()

    async def convert(
        self,
        source_audio_path: str,
        target_audio_path: str,
        params: ConcurrentInferenceParams,
    ) -> Tuple[str, int, np.ndarray]:
        request_id = str(uuid.uuid4())
        total_started_at = time.perf_counter()
        prepare_started_at = time.perf_counter()
        source_features, timbre_features, feature_lock_wait_sec = await self._prepare_features_async(
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
            ar_chunks = self._build_ar_chunks(request_id, source_features, timbre_features, params)
            ar_requests = [
                ARGenerateRequest(
                    request_id=request_id,
                    chunk_index=chunk_index,
                    prompt_text=prompt_text,
                    prompt_target=prompt_target,
                    params=params,
                )
                for chunk_index, (prompt_text, prompt_target) in enumerate(ar_chunks)
            ]
            ar_outputs = await asyncio.gather(*(
                self.ar_scheduler.submit(ar_request) for ar_request in ar_requests
            ))
            ar_sec = time.perf_counter() - ar_started_at
            generated_tokens = sum(int(output.size(-1)) for output in ar_outputs)
            logger.info(
                "request=%s stage=ar_done ar_sec=%.3f chunks=%s generated_tokens=%s tokens_per_sec=%.2f",
                request_id,
                ar_sec,
                len(ar_outputs),
                generated_tokens,
                generated_tokens / ar_sec if ar_sec > 0 else 0.0,
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

    async def _prepare_features_async(
        self,
        request_id: str,
        source_audio_path: str,
        target_audio_path: str,
        require_source_narrow: bool,
    ) -> Tuple[SourceFeatures, TimbreFeatures, float]:
        timbre_started_at = time.perf_counter()
        feature_lock_wait_started_at = time.perf_counter()
        async with self._feature_lock:
            feature_lock_wait_sec = time.perf_counter() - feature_lock_wait_started_at
            timbre = await asyncio.to_thread(
                self._get_or_compute_timbre_features,
                request_id,
                target_audio_path,
            )
        logger.info(
            "request=%s stage=timbre_done elapsed_sec=%.3f cache_key=%s",
            request_id,
            time.perf_counter() - timbre_started_at,
            timbre.cache_key,
        )

        source_started_at = time.perf_counter()
        source = await asyncio.to_thread(
            self._compute_source_features,
            request_id,
            source_audio_path,
            require_source_narrow,
        )
        logger.info(
            "request=%s stage=source_features_done elapsed_sec=%.3f source_mel_len=%s require_narrow=%s",
            request_id,
            time.perf_counter() - source_started_at,
            source.source_mel_len,
            require_source_narrow,
        )
        return source, timbre, feature_lock_wait_sec

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
        source = self._compute_source_features(request_id, source_audio_path, require_source_narrow)
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
        features = self._compute_timbre_features(request_id, cache_key, target_audio_path)
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
    def _compute_timbre_features(self, request_id: str, cache_key: str, target_audio_path: str) -> TimbreFeatures:
        load_started_at = time.perf_counter()
        target_wave = librosa.load(target_audio_path, sr=self.vc_wrapper.sr)[0]
        target_wave = target_wave[: self.vc_wrapper.sr * (self.vc_wrapper.dit_max_context_len - 5)]
        load_sec = time.perf_counter() - load_started_at
        tensor_started_at = time.perf_counter()
        target_wave_tensor = torch.tensor(target_wave).unsqueeze(0).float().to(self.device)
        self._synchronize_for_profiling()
        tensor_to_device_sec = time.perf_counter() - tensor_started_at
        resample_started_at = time.perf_counter()
        target_wave_16k = librosa.resample(target_wave, orig_sr=self.vc_wrapper.sr, target_sr=16000)
        resample_sec = time.perf_counter() - resample_started_at
        tensor_16k_started_at = time.perf_counter()
        target_wave_16k_tensor = torch.tensor(target_wave_16k).unsqueeze(0).to(self.device)
        self._synchronize_for_profiling()
        tensor_16k_to_device_sec = time.perf_counter() - tensor_16k_started_at

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            mel_started_at = time.perf_counter()
            self._synchronize_for_profiling()
            target_mel = self.vc_wrapper.mel_fn(target_wave_tensor)
            self._synchronize_for_profiling()
            mel_sec = time.perf_counter() - mel_started_at
            target_mel_len = target_mel.size(2)
            wide_started_at = time.perf_counter()
            self._synchronize_for_profiling()
            target_content_indices = self.vc_wrapper._process_content_features(target_wave_16k_tensor, is_narrow=False)
            self._synchronize_for_profiling()
            wide_content_sec = time.perf_counter() - wide_started_at
            narrow_started_at = time.perf_counter()
            self._synchronize_for_profiling()
            target_narrow_indices = self.vc_wrapper._process_content_features(target_wave_16k_tensor, is_narrow=True)
            self._synchronize_for_profiling()
            narrow_content_sec = time.perf_counter() - narrow_started_at
            reduction_started_at = time.perf_counter()
            target_narrow_reduced, _ = self.vc_wrapper.duration_reduction_func(target_narrow_indices[0], 1)
            reduction_sec = time.perf_counter() - reduction_started_at
            style_started_at = time.perf_counter()
            self._synchronize_for_profiling()
            target_style = self.vc_wrapper.compute_style(target_wave_16k_tensor)
            self._synchronize_for_profiling()
            style_sec = time.perf_counter() - style_started_at
            prompt_started_at = time.perf_counter()
            self._synchronize_for_profiling()
            prompt_condition, _ = self.vc_wrapper.cfm_length_regulator(
                target_content_indices,
                ylens=torch.LongTensor([target_mel_len]).to(self.device),
            )
            self._synchronize_for_profiling()
            prompt_regulator_sec = time.perf_counter() - prompt_started_at

        if self.enable_profiling:
            logger.info(
                (
                    "request=%s stage=timbre_profile cache_key=%s load_sec=%.3f tensor_to_device_sec=%.3f "
                    "resample_sec=%.3f tensor_16k_to_device_sec=%.3f mel_sec=%.3f "
                    "wide_content_sec=%.3f narrow_content_sec=%.3f reduction_sec=%.3f "
                    "style_sec=%.3f prompt_regulator_sec=%.3f target_mel_len=%s"
                ),
                request_id,
                cache_key,
                load_sec,
                tensor_to_device_sec,
                resample_sec,
                tensor_16k_to_device_sec,
                mel_sec,
                wide_content_sec,
                narrow_content_sec,
                reduction_sec,
                style_sec,
                prompt_regulator_sec,
                target_mel_len,
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
    def _compute_source_features(self, request_id: str, source_audio_path: str, require_narrow: bool) -> SourceFeatures:
        load_started_at = time.perf_counter()
        source_wave = librosa.load(source_audio_path, sr=self.vc_wrapper.sr)[0]
        load_sec = time.perf_counter() - load_started_at
        tensor_started_at = time.perf_counter()
        source_wave_tensor = torch.tensor(source_wave).unsqueeze(0).float().to(self.device)
        self._synchronize_for_profiling()
        tensor_to_device_sec = time.perf_counter() - tensor_started_at
        resample_started_at = time.perf_counter()
        source_wave_16k = librosa.resample(source_wave, orig_sr=self.vc_wrapper.sr, target_sr=16000)
        resample_sec = time.perf_counter() - resample_started_at
        tensor_16k_started_at = time.perf_counter()
        source_wave_16k_tensor = torch.tensor(source_wave_16k).unsqueeze(0).to(self.device)
        self._synchronize_for_profiling()
        tensor_16k_to_device_sec = time.perf_counter() - tensor_16k_started_at

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            mel_started_at = time.perf_counter()
            self._synchronize_for_profiling()
            source_mel = self.vc_wrapper.mel_fn(source_wave_tensor)
            self._synchronize_for_profiling()
            mel_sec = time.perf_counter() - mel_started_at
            wide_started_at = time.perf_counter()
            self._synchronize_for_profiling()
            source_content_indices = self.vc_wrapper._process_content_features(source_wave_16k_tensor, is_narrow=False)
            self._synchronize_for_profiling()
            wide_content_sec = time.perf_counter() - wide_started_at
            source_narrow_reduced = None
            narrow_content_sec = 0.0
            reduction_sec = 0.0
            if require_narrow:
                narrow_started_at = time.perf_counter()
                self._synchronize_for_profiling()
                source_narrow_indices = self.vc_wrapper._process_content_features(source_wave_16k_tensor, is_narrow=True)
                self._synchronize_for_profiling()
                narrow_content_sec = time.perf_counter() - narrow_started_at
                reduction_started_at = time.perf_counter()
                source_narrow_reduced, _ = self.vc_wrapper.duration_reduction_func(source_narrow_indices[0], 1)
                reduction_sec = time.perf_counter() - reduction_started_at

        if self.enable_profiling:
            logger.info(
                (
                    "request=%s stage=source_profile load_sec=%.3f tensor_to_device_sec=%.3f "
                    "resample_sec=%.3f tensor_16k_to_device_sec=%.3f mel_sec=%.3f "
                    "wide_content_sec=%.3f narrow_content_sec=%.3f reduction_sec=%.3f "
                    "source_mel_len=%s require_narrow=%s"
                ),
                request_id,
                load_sec,
                tensor_to_device_sec,
                resample_sec,
                tensor_16k_to_device_sec,
                mel_sec,
                wide_content_sec,
                narrow_content_sec,
                reduction_sec,
                source_mel.size(2),
                require_narrow,
            )

        return SourceFeatures(
            source_audio_path=source_audio_path,
            source_mel_len=source_mel.size(2),
            source_content_indices=source_content_indices,
            source_narrow_reduced=source_narrow_reduced,
        )

    def _style_cfm_source_token_budget(
        self,
        source: SourceFeatures,
        timbre: TimbreFeatures,
        params: ConcurrentInferenceParams,
    ) -> int:
        if source.source_narrow_reduced is None or source.source_narrow_reduced.numel() <= 0:
            return 1

        max_condition_len = int(self.vc_wrapper.cfm_compile_buckets[-1])
        source_frame_budget = max(1, max_condition_len - int(timbre.target_mel_len) - CFM_CHUNK_SAFETY_FRAMES)
        frames_per_narrow_token = max(
            float(source.source_mel_len) / max(1, int(source.source_narrow_reduced.numel())),
            1e-6,
        )
        length_adjust = max(float(params.length_adjust), 1e-6)
        return max(1, int(source_frame_budget / frames_per_narrow_token / length_adjust))

    def _build_ar_prompt_text(
        self,
        target_prefix: torch.Tensor,
        source_chunk: torch.Tensor,
        anonymization_only: bool,
    ) -> torch.Tensor:
        if anonymization_only:
            ar_tokens = source_chunk
        else:
            ar_tokens = torch.cat([target_prefix, source_chunk], dim=0)
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                return self.vc_wrapper.ar_length_regulator(ar_tokens[None])[0]

    def _ar_prompt_sequence_len(self, prompt_text: torch.Tensor, prompt_target: torch.Tensor) -> int:
        # build_generation_inputs inserts two separator embeddings around prompt_text.
        return int(prompt_text.size(1)) + int(prompt_target.size(-1)) + 2

    def _ar_chunk_has_decode_room(
        self,
        prompt_text: torch.Tensor,
        prompt_target: torch.Tensor,
        source_chunk_len: int,
    ) -> bool:
        prompt_seq_len = self._ar_prompt_sequence_len(prompt_text, prompt_target)
        generation_reserve = max(
            int(self.ar_scheduler.min_tokens_before_eos) + AR_DECODE_SAFETY_TOKENS,
            int(source_chunk_len) + AR_DECODE_SAFETY_TOKENS,
        )
        return prompt_seq_len + generation_reserve < int(self.ar_scheduler.max_seq_len)

    def _fit_ar_target_context(
        self,
        request_id: str,
        source_tokens: torch.Tensor,
        target_prefix: torch.Tensor,
        prompt_target: torch.Tensor,
        anonymization_only: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if anonymization_only or source_tokens.numel() <= 0:
            return target_prefix, prompt_target

        original_prefix_len = int(target_prefix.numel())
        original_prompt_target_len = int(prompt_target.size(-1))
        min_decode_room = int(self.ar_scheduler.min_tokens_before_eos) + AR_DECODE_SAFETY_TOKENS
        first_source_token = source_tokens[:1]

        while True:
            prompt_text = self._build_ar_prompt_text(target_prefix, first_source_token, anonymization_only=False)
            prompt_seq_len = self._ar_prompt_sequence_len(prompt_text, prompt_target)
            if prompt_seq_len + min_decode_room < int(self.ar_scheduler.max_seq_len):
                break

            overflow = prompt_seq_len + min_decode_room - int(self.ar_scheduler.max_seq_len) + 1
            if int(prompt_target.size(-1)) > 0:
                trim = min(int(prompt_target.size(-1)), max(1, overflow))
                prompt_target = prompt_target[:, : int(prompt_target.size(-1)) - trim]
                continue
            if int(target_prefix.numel()) > 0:
                trim = min(int(target_prefix.numel()), max(1, overflow))
                target_prefix = target_prefix[: int(target_prefix.numel()) - trim]
                continue
            break

        if original_prefix_len != int(target_prefix.numel()) or original_prompt_target_len != int(prompt_target.size(-1)):
            logger.warning(
                (
                    "request=%s stage=ar_target_context_trimmed original_target_prefix_len=%s "
                    "target_prefix_len=%s original_prompt_target_len=%s prompt_target_len=%s max_seq_len=%s"
                ),
                request_id,
                original_prefix_len,
                int(target_prefix.numel()),
                original_prompt_target_len,
                int(prompt_target.size(-1)),
                int(self.ar_scheduler.max_seq_len),
            )
        return target_prefix, prompt_target

    def _fit_ar_chunk(
        self,
        target_prefix: torch.Tensor,
        prompt_target: torch.Tensor,
        source_tokens: torch.Tensor,
        start: int,
        max_chunk_size: int,
        anonymization_only: bool,
    ) -> Tuple[int, torch.Tensor]:
        low = 1
        high = max(1, int(max_chunk_size))
        best_size = 0
        best_prompt_text = None

        while low <= high:
            mid = (low + high) // 2
            source_chunk = source_tokens[start:start + mid]
            prompt_text = self._build_ar_prompt_text(target_prefix, source_chunk, anonymization_only)
            if self._ar_chunk_has_decode_room(prompt_text, prompt_target, int(source_chunk.numel())):
                best_size = mid
                best_prompt_text = prompt_text
                low = mid + 1
            else:
                high = mid - 1

        if best_prompt_text is not None:
            return best_size, best_prompt_text

        source_chunk = source_tokens[start:start + 1]
        prompt_text = self._build_ar_prompt_text(target_prefix, source_chunk, anonymization_only)
        prompt_seq_len = self._ar_prompt_sequence_len(prompt_text, prompt_target)
        if prompt_seq_len < int(self.ar_scheduler.max_seq_len):
            logger.warning(
                (
                    "stage=ar_chunk_decode_room_low prompt_seq_len=%s max_seq_len=%s "
                    "source_chunk_len=1"
                ),
                prompt_seq_len,
                int(self.ar_scheduler.max_seq_len),
            )
            return 1, prompt_text

        raise RuntimeError(
            f"AR prompt is too long even after adaptive chunking: "
            f"{prompt_seq_len} >= max_seq_len {self.ar_scheduler.max_seq_len}"
        )

    def _synchronize_for_profiling(self) -> None:
        if self.enable_profiling:
            synchronize_device(self.device)

    def _build_ar_chunks(
        self,
        request_id: str,
        source: SourceFeatures,
        timbre: TimbreFeatures,
        params: ConcurrentInferenceParams,
    ) -> Sequence[Tuple[torch.Tensor, torch.Tensor]]:
        if source.source_narrow_reduced is None:
            raise RuntimeError("source narrow features are required for convert_style=True")

        if params.anonymization_only:
            prefix_len = 0
            prompt_target = torch.zeros([1, 0], dtype=torch.long, device=self.device)
            target_prefix = source.source_narrow_reduced[:0]
        else:
            target_prefix = timbre.target_narrow_reduced
            prompt_target = timbre.target_content_indices

        target_prefix, prompt_target = self._fit_ar_target_context(
            request_id,
            source.source_narrow_reduced,
            target_prefix,
            prompt_target,
            params.anonymization_only,
        )
        prefix_len = 0 if params.anonymization_only else int(target_prefix.numel())
        cfm_chunk_budget = self._style_cfm_source_token_budget(source, timbre, params)
        max_chunk_size = max(1, min(self.vc_wrapper.ar_max_content_len - prefix_len, cfm_chunk_budget))

        chunks = []
        start = 0
        source_token_count = int(source.source_narrow_reduced.numel())
        while start < source_token_count:
            candidate_size = min(max_chunk_size, source_token_count - start)
            chunk_size, prompt_text = self._fit_ar_chunk(
                target_prefix,
                prompt_target,
                source.source_narrow_reduced,
                start,
                candidate_size,
                params.anonymization_only,
            )
            chunks.append((prompt_text, prompt_target))
            start += chunk_size
        logger.info(
            (
                "request=%s stage=ar_chunks_prepared chunks=%s source_narrow_len=%s target_prefix_len=%s "
                "max_chunk_size=%s cfm_chunk_budget=%s prompt_target_len=%s max_seq_len=%s anonymization_only=%s"
            ),
            request_id,
            len(chunks),
            source_token_count,
            prefix_len,
            max_chunk_size,
            cfm_chunk_budget,
            int(prompt_target.size(-1)),
            int(self.ar_scheduler.max_seq_len),
            params.anonymization_only,
        )
        return chunks

    def metrics(self) -> Dict[str, Dict[str, float]]:
        return {
            "ar_scheduler": self.ar_scheduler.metrics(),
            "cfm_scheduler": self.cfm_scheduler.metrics(),
            "timbre_cache": self.timbre_cache.metrics(),
        }
