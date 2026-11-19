from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, List, Optional
import heapq
import bisect
from concurrent.futures import ThreadPoolExecutor
import torch
from sglang.srt.managers.ttft_probe import probe as ttft_probe
import os, copy
from typing import Any
import hashlib

from sglang.srt.managers.cache_controller import HiCacheController, PrefetchOperation
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams, MatchResult, BasePrefixCache, EvictParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)
from sglang.srt.mem_cache.radix_cache import (
    RadixCache,
    RadixKey,
    compute_node_hash_values,
    split_node_hash_value,
)
from sglang.srt.observability.metrics_collector import StorageMetricsCollector
from sglang.srt.utils import bind_to_closest_numa_node_cuda

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs

from sglang.srt.mem_cache.evict_policy import (
    EvictionStrategy,
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
)
from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

import hashlib


class HitCacheNode:
    def __init__(
        self,
        value: torch.Tensor,
        original_position: torch.Tensor,
        current_position: torch.Tensor,
    ):
        self.original_position: torch.Tensor = original_position
        self.current_position: torch.Tensor = current_position
        self.value: torch.Tensor = value

class ChunkNode:

    counter = 0

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        self.prefix_text: str = ""
        self.text_without_prefix: str = ""
        self.cache_prefix_token_len : int = 0
        self.values: Optional[List[torch.Tensor]] = None
        self.evictable_values: Optional[List[torch.Tensor]] = None
        self.key: Optional[List[int]] = None
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()

        self.hit_count = 0
        # indicating the node is locked to protect from eviction
        # incremented when the node is referenced by a storage operation
        self.host_ref_counter = 1 # Hold a host reference from allocation until the first consumer releases it.
        # store the host indices of KV cache
        self.host_value: Optional[torch.Tensor] = None
        self.disk_tensor_path: Optional[Path] = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None
        # priority for priority-aware eviction
        self.priority = priority
        self.is_preprocess_cache = False ## false is raw cache, true is preprocess cache
        self.preprocess_cache_key = ""
        self.text_without_prefix_ids: List[int] = []

        self.id = ChunkNode.counter if id is None else id
        ChunkNode.counter += 1

    @property
    def evicted(self):
        return len(self.values) == 0

    @property
    def num_tokens(self) -> int:
        if self.host_value is not None:
            return len(self.host_value)
        return len(self.text_without_prefix_ids)

    @property
    def backuped(self):
        return self.host_value is not None

    def protect_host(self):
        """Protect the host value from eviction."""
        self.host_ref_counter += 1

    def hash_key(self) -> str:
        text = f"is_preprocess_cache={self.is_preprocess_cache} text_without_prefix={self.text_without_prefix}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def release_host(self):
        """Release the host value, allowing it to be evicted."""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    # @lru_cache(maxsize=1)
    # def get_prefix_hash_values(self, node: TreeNode) -> List[str]:
    #     if node is None or node.hash_value is None:
    #         return []
    #
    #     return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


class FusionragCache(RadixCache):
    def init_metrics_collector(self):
        super().init_metrics_collector(metric_prefix="sglang:fusionrag_")


    def __init__(self, params: CacheInitParams, server_args: ServerArgs, tp_rank: int, tp_size: int):
        if server_args.hicache_io_backend == "direct":
            # FIXME: move this logic into server_args parsing
            if server_args.hicache_mem_layout == "page_first":
                server_args.hicache_mem_layout = "page_first_direct"
                logger.warning(
                    "Page first layout is not supported with direct IO backend, switching to page first direct layout"
                )

        if not server_args.disable_hicache_numa_detect:
            bind_to_closest_numa_node_cuda()


        self.page_size = params.page_size
        self.kv_cache = params.token_to_kv_pool_allocator.get_kvcache()
        self.use_qwen = False
        self.use_deepseek = False

        cache_size = server_args.fusionrag_cache_size
        if cache_size == 0:
            cache_size = server_args.hicache_size

        print(f"FusionragCache tp_rank = {tp_rank}, cache_size={cache_size}")

        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = MHATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                cache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
            self.use_qwen = True
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                cache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
            self.use_deepseek = True
        else:
            raise ValueError(f"HiRadixCache only supports MHA and MLA yet")

        self.tp_group = params.tp_cache_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.pp_rank = params.pp_rank
        self.pp_size = params.pp_size
        self.enable_storage = server_args.hicache_storage_backend is not None
        self.enable_storage_metrics = self.enable_storage and params.enable_metrics

        (
            extra_config,
            prefetch_threshold,
            prefetch_timeout_base,
            prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys,
        ) = self._parse_storage_backend_extra_config(
            server_args.hicache_storage_backend_extra_config
        )
        self.prefetch_threshold = prefetch_threshold
        self.prefetch_timeout_base = prefetch_timeout_base
        self.prefetch_timeout_per_page = (
            self.page_size / 1024 * prefetch_timeout_per_ki_token
        )
        self.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys
        # TODO: support more timeout check functions

        self.load_cache_event = threading.Event()
        self.cache_controller = HiCacheController(
            params.token_to_kv_pool_allocator,
            self.token_to_kv_pool_host,
            self.page_size,
            self.tp_group,
            load_cache_event=self.load_cache_event,
            write_policy=server_args.hicache_write_policy,
            io_backend=server_args.hicache_io_backend,
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=self.prefetch_threshold,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=extra_config,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
        )
        if self.enable_storage_metrics:
            # TODO: support pp
            labels = {
                "storage_backend": server_args.hicache_storage_backend,
                "tp_rank": self.cache_controller.tp_rank,
                "dp_rank": self.cache_controller.dp_rank,
                "pp_rank": self.cache_controller.pp_rank,
                "pp_size": self.cache_controller.pp_size,
            }
            self.storage_metrics_collector = StorageMetricsCollector(labels=labels)

        # record the nodes with ongoing write through
        self.ongoing_write_through = {}
        # record the node segments with ongoing load back
        self.ongoing_load_back = {}
        # record the ongoing prefetch requests
        self.ongoing_prefetch = {}
        self.ongoing_backup = {}
        # todo: dynamically adjust the threshold
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 10
        self.all_nodes: List[ChunkNode] = []
        self.key_to_nodes = {}
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.eviction_policy = "lru" ## cache_debug
        self.eviction_strategy: EvictionStrategy = LRUStrategy()

        self.document_cache_dir = None
        self.ssd_restore_samples = []
        self.storage_read_mode = "buffered"
        self.document_prefetch_enabled = False
        self.document_staging_enabled = False
        self.document_staging_indices = {}
        self.document_prefetch_jobs_limit = 64
        self.document_staging_token_limit = 8192
        self.document_prefetch_workers = 2
        self.document_staging_cancelled = 0
        self.document_prefetch_executor = None
        self.document_prefetch_jobs = {}
        self.document_prefetch_thread = threading.local()
        self.document_prefetch_counts = dict(submitted=0, ready=0, running=0, missing=0, failed=0)
        storage_root = os.environ.get("SGLANG_FUSIONRAG_CACHE_DIR")
        self.replicated_document_kv = os.environ.get("SGLANG_FUSIONRAG_REPLICATED_MLA") == "1"
        self.document_data_dir = None
        self.document_dram_dir = os.environ.get("SGLANG_FUSIONRAG_DRAM_CACHE_DIR")
        if self.replicated_document_kv and (
            not storage_root
            or Path(server_args.model_path).name != "Kimi-K2.6"
            or type(self.kv_cache) is not MLATokenToKVPool
            or self.token_to_kv_pool_host.dtype != torch.bfloat16
            or self.tp_world_size != tp_size
            or server_args.nnodes != 1
            or server_args.pp_size != 1
            or server_args.dp_size != 1
            or server_args.enable_dp_attention
        ):
            raise ValueError("Shared document KV requires verified replicated Kimi-K2.6 BF16 MLA")
        if storage_root:
            host_pool = self.token_to_kv_pool_host
            if not self.use_deepseek or host_pool.layout != "layer_first":
                raise ValueError("Persistent document KV requires a layer-first MLA host pool")
            identity = json.dumps(
                {
                    "model_path": os.path.realpath(server_args.model_path),
                    "revision": server_args.revision,
                    "dtype": str(host_pool.dtype),
                    "layer_count": host_pool.layer_num,
                    "kv_width": host_pool.kv_cache_dim,
                },
                sort_keys=True,
            )
            namespace = hashlib.sha256(identity.encode()).hexdigest()
            layout = f"tp{tp_size}-replicated-mla" if self.replicated_document_kv else f"tp{tp_size}-rank{tp_rank}"
            self.document_cache_dir = Path(storage_root) / "v1" / namespace / layout
            self.document_data_dir = self.document_cache_dir
            self.document_cache_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(params=params)
        if self.document_cache_dir is not None:
            self._index_document_cache()

    def _index_document_cache(self) -> None:
        for entry in self.document_cache_dir.iterdir():
            if entry.name.startswith(".") or not entry.is_dir():
                continue
            with (entry / "metadata.json").open(encoding="utf-8") as stream:
                metadata = json.load(stream)
            node = ChunkNode()
            node.text_without_prefix = metadata["text"]
            node.prefix_text = metadata["prefix_text"]
            node.cache_prefix_token_len = metadata["prefix_token_count"]
            node.text_without_prefix_ids = metadata["token_ids"]
            node.values = []
            node.host_ref_counter = 0
            node.disk_tensor_path = entry / "kv.pt"
            if entry.name != node.hash_key() or not node.disk_tensor_path.is_file():
                raise RuntimeError(f"Invalid FusionRAG document KV entry: {entry}")
            self.key_to_nodes[entry.name] = node
            self.all_nodes.append(node)

        if self.tp_world_size > 1:
            keys_by_rank = [None] * self.tp_world_size
            torch.distributed.all_gather_object(
                keys_by_rank, list(self.key_to_nodes), group=self.tp_group
            )
            complete = set(keys_by_rank[0]).intersection(*keys_by_rank[1:])
            incomplete = self.key_to_nodes.keys() - complete
            if incomplete:
                logger.warning("Ignoring %d document KV blocks missing TP shards", len(incomplete))
                self.key_to_nodes = {
                    key: node for key, node in self.key_to_nodes.items() if key in complete
                }
                self.all_nodes = list(self.key_to_nodes.values())

        self.all_nodes.sort(key=lambda node: len(node.text_without_prefix), reverse=True)
        logger.info(
            "Indexed %d persisted FusionRAG document KV blocks on TP rank %d",
            len(self.all_nodes),
            self.tp_rank,
        )

    def _persist_document(self, node: ChunkNode) -> None:
        destination = self.document_cache_dir / node.hash_key()
        if self.replicated_document_kv and self.tp_rank != 0:
            node.disk_tensor_path = destination / "kv.pt"
            return
        if not destination.exists():
            host_pool = self.token_to_kv_pool_host
            metadata = {
                "text": node.text_without_prefix,
                "prefix_text": node.prefix_text,
                "prefix_token_count": node.cache_prefix_token_len,
                "token_ids": node.text_without_prefix_ids,
            }
            with TemporaryDirectory(prefix=".pending-", dir=self.document_cache_dir) as scratch:
                scratch = Path(scratch)
                with (scratch / "kv.pt").open("wb") as stream:
                    torch.save(host_pool.kv_buffer.index_select(1, node.host_value), stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                with (scratch / "metadata.json").open("w", encoding="utf-8") as stream:
                    json.dump(metadata, stream, ensure_ascii=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                self._sync_directory(scratch)
                scratch.rename(destination)
            self._sync_directory(self.document_cache_dir)
        node.disk_tensor_path = destination / "kv.pt"

    @staticmethod
    def _sync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def set_document_source(self, tier: str) -> None:
        if not self.replicated_document_kv:
            raise RuntimeError("The document tier control requires replicated persistent KV")
        if tier == "ssd":
            self.document_data_dir = self.document_cache_dir
            return
        if tier != "dram" or self.document_dram_dir is None:
            raise ValueError("Unknown or unavailable document KV source")
        shared_path = Path(self.document_dram_dir) / self.document_cache_dir.name
        if not shared_path.is_dir():
            raise RuntimeError("The in-memory document store is not ready")
        self.document_data_dir = shared_path

    def _document_payload(self, path: Path) -> Path:
        return self.document_data_dir / path.relative_to(self.document_cache_dir)

    def _read_document_file(self, path):
        # Touch file pages in a worker. The scheduler owns pinned memory and publishes nodes.
        buffer = getattr(self.document_prefetch_thread, "buffer", None)
        if buffer is None:
            buffer = bytearray(1024 * 1024)
            self.document_prefetch_thread.buffer = buffer
        with self._document_payload(path).open("rb", buffering=0) as stream:
            while stream.readinto(buffer):
                pass

    def _recyclable_document_slots(self, protected):
        return iter(sorted(
            (node for key, node in self.key_to_nodes.items()
             if key not in protected and node.host_value is not None
             and node.disk_tensor_path is not None and node.host_ref_counter == 0 and not node.values),
            key=lambda node: node.hit_count,
        ))

    def prefetch_request_document_files(self, req):
        if (
            not self.document_prefetch_enabled or self.document_cache_dir is None
            or req.is_kv_gen or req.clear_cache or req.use_preprocess_cache
            or req.cache_is_preprocess_list is not None or not req.prefix_prompt_list
        ):
            return
        jobs = self.document_prefetch_jobs
        remaining = self.document_prefetch_jobs_limit - sum(not future.done() for future in jobs.values())
        staged = self.document_staging_indices
        host_pool = self.token_to_kv_pool_host
        staging = self.document_staging_enabled and self.storage_read_mode == "mapped"
        if staging:
            remaining = min(remaining, self.document_prefetch_jobs_limit - len(staged))
            token_budget = self.document_staging_token_limit - sum(len(indices) for indices in staged.values())
        recyclable = None
        for text in req.prefix_prompt_list:
            if remaining <= 0:
                break
            key = hashlib.sha256(f"is_preprocess_cache=False text_without_prefix={text}".encode()).hexdigest()
            node = self.key_to_nodes.get(key)
            if node is None:
                break
            path = node.disk_tensor_path
            if node.host_value is not None or path is None or path in jobs:
                continue
            if self.document_prefetch_executor is None:
                self.document_prefetch_executor = ThreadPoolExecutor(max_workers=self.document_prefetch_workers)
            if staging:
                if node.num_tokens > token_budget:
                    continue
                if self.replicated_document_kv and host_pool.available_size() < node.num_tokens:
                    if recyclable is None:
                        protected = {
                            hashlib.sha256(f"is_preprocess_cache=False text_without_prefix={part}".encode()).hexdigest()
                            for part in req.prefix_prompt_list
                        }
                        recyclable = self._recyclable_document_slots(protected)
                    while host_pool.available_size() < node.num_tokens:
                        victim = next(recyclable, None)
                        if victim is None:
                            break
                        host_pool.free(victim.host_value)
                        victim.host_value = None
                indices = host_pool.alloc(node.num_tokens)
                if indices is None:
                    break
                try:
                    jobs[path] = self.document_prefetch_executor.submit(
                        self._copy_document_to_host, path, node.num_tokens, indices, True, True
                    )
                except Exception:
                    host_pool.free(indices)
                    raise
                staged[path] = indices
                token_budget -= node.num_tokens
            else:
                jobs[path] = self.document_prefetch_executor.submit(self._read_document_file, path)
            remaining -= 1
            self.document_prefetch_counts["submitted"] += 1

    def _observe_document_prefetch(self, nodes):
        counts = self.document_prefetch_counts
        for node in nodes:
            future = self.document_prefetch_jobs.get(node.disk_tensor_path)
            if future is None:
                counts["missing"] += 1
            elif not future.done():
                counts["running"] += 1
            elif future.cancelled() or future.exception() is not None:
                counts["failed"] += 1
            else:
                counts["ready"] += 1

    def _reset_document_prefetch(self):
        # A restore can consume a file while the background read is still in flight.
        if self.document_prefetch_executor is not None:
            self.document_prefetch_executor.shutdown(wait=True, cancel_futures=True)
            self.document_prefetch_executor = None
        self.document_prefetch_jobs.clear()
        self.document_staging_indices.clear()
        self.document_staging_cancelled = 0
        for key in self.document_prefetch_counts:
            self.document_prefetch_counts[key] = 0

    @torch.inference_mode()
    def _copy_document_to_host(self, path, num_tokens, indices, mapped, background=False):
        copy_log = ttft_probe.copies
        if copy_log is not None:
            started = time.perf_counter()
        host_pool = self.token_to_kv_pool_host
        payload = self._document_payload(path)
        tensor = torch.load(payload, map_location="cpu", weights_only=True, mmap=mapped)
        expected = (host_pool.layer_num, num_tokens, 1, host_pool.kv_cache_dim)
        if tuple(tensor.shape) != expected or tensor.dtype != host_pool.dtype:
            raise RuntimeError(f"Incompatible FusionRAG KV tensor: {path}")
        if copy_log is not None:
            mapped_at = time.perf_counter()
        start = int(indices[0])
        if mapped and torch.equal(indices, torch.arange(start, start + len(indices))):
            host_pool.kv_buffer.narrow(1, start, len(indices)).copy_(tensor)
        else:
            host_pool.set_from_indices(indices, tensor)
        if copy_log is not None:
            copy_log.append([started, mapped_at, time.perf_counter(), num_tokens, background])

    def _release_staged_documents(self, needed, protected_paths):
        host_pool = self.token_to_kv_pool_host
        for path, indices in list(self.document_staging_indices.items()):
            if host_pool.available_size() >= needed:
                break
            if path in protected_paths:
                continue
            future = self.document_prefetch_jobs.pop(path)
            if not future.cancel():
                # Wait for any writer before returning its host indices.
                future.exception()
            del self.document_staging_indices[path]
            host_pool.free(indices)

    def _restore_document_chunks(self, nodes: List[ChunkNode]) -> None:
        cold_nodes = [node for node in dict.fromkeys(nodes) if node.host_value is None]
        if self.document_prefetch_enabled:
            self._observe_document_prefetch(cold_nodes)
        if not cold_nodes:
            return
        host_pool = self.token_to_kv_pool_host
        required = sum(node.num_tokens for node in cold_nodes)
        staged = self.document_staging_indices
        needed = sum(node.num_tokens for node in cold_nodes if node.disk_tensor_path not in staged) if staged else required
        shortage = needed - host_pool.available_size()
        if shortage > 0:
            self.evict_host(shortage, exclude=set(nodes))
        if needed > host_pool.available_size() and staged:
            self._release_staged_documents(needed, {node.disk_tensor_path for node in cold_nodes})
        if needed > host_pool.available_size():
            raise RuntimeError("Insufficient FusionRAG host KV slots for request")

        started = time.perf_counter()
        with torch.profiler.record_function(f"fusionrag_ssd_to_dram:{required}"):
            restored = []
            try:
                for node in cold_nodes:
                    path = node.disk_tensor_path
                    indices = staged.pop(path, None)
                    future = None
                    if indices is None:
                        indices = host_pool.alloc(node.num_tokens)
                        if indices is None:
                            raise RuntimeError("Insufficient FusionRAG host KV slots for document")
                    else:
                        future = self.document_prefetch_jobs.pop(path)
                    restored.append((node, indices))
                    if future is not None:
                        if future.cancel():
                            self.document_staging_cancelled += 1
                        elif future.exception() is None:
                            continue
                    self._copy_document_to_host(
                        path, node.num_tokens, indices, self.storage_read_mode == "mapped"
                    )
            except Exception:
                for _, indices in restored:
                    host_pool.free(indices)
                raise
            for node, indices in restored:
                node.host_value = indices
        if os.environ.get("FUSIONRAG_STORAGE_BENCH") == "1":
            ended = time.perf_counter()
            self.ssd_restore_samples.append((len(cold_nodes), required, (ended - started) * 1000))
            ttft_probe.restore(started, ended, required, len(cold_nodes))

    def storage_bench_state(self):
        return {
            "indexed_blocks": len(self.all_nodes),
            "replicated_document_kv": self.replicated_document_kv,
            "document_source": "ssd" if self.document_data_dir == self.document_cache_dir else "dram",
            "read_mode": self.storage_read_mode,
            "document_prefetch_enabled": self.document_prefetch_enabled,
            "document_staging_enabled": self.document_staging_enabled,
            "document_prefetch_limits": [self.document_prefetch_workers, self.document_prefetch_jobs_limit,
                                         self.document_staging_token_limit],
            "document_staging_cancelled": self.document_staging_cancelled,
            "document_prefetch": self.document_prefetch_counts.copy(),
            "resident_tokens": sum(node.num_tokens for node in self.all_nodes if node.host_value is not None),
            "restore_samples": self.ssd_restore_samples.copy(),
        }

    def reset_storage_bench(self):
        if self.document_cache_dir is None:
            raise RuntimeError("Document SSD persistence is required")
        self.reset()
        self.ssd_restore_samples.clear()
        self._index_document_cache()

    def _parse_storage_backend_extra_config(
        self, storage_backend_extra_config: Optional[str]
    ):
        """
        Parse storage backend extra config JSON and extract specific parameters.

        Args:
            storage_backend_extra_config: JSON string containing extra configuration

        Returns:
            tuple: (extra_config_dict, prefetch_threshold, prefetch_timeout_base, prefetch_timeout_per_ki_token, hicache_storage_pass_prefix_keys)
        """
        # Parse extra config JSON if provided
        extra_config = {}
        if storage_backend_extra_config:
            try:
                extra_config = json.loads(storage_backend_extra_config)
            except Exception as e:
                logger.error(f"Invalid backend extra config JSON: {e}")
                raise e

        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)  # tokens
        prefetch_timeout_base = extra_config.pop("prefetch_timeout_base", 1)  # seconds
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", 0.25
        )  # seconds per 1024 tokens
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )

        if not isinstance(prefetch_threshold, int):
            raise ValueError(
                f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}"
            )
        if not isinstance(prefetch_timeout_base, (int, float)):
            raise ValueError(
                f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}"
            )
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)):
            raise ValueError(
                f"prefetch_timeout_per_ki_token must be number, got {type(prefetch_timeout_per_ki_token).__name__}"
            )

        return (
            extra_config,
            prefetch_threshold,
            float(prefetch_timeout_base),
            float(prefetch_timeout_per_ki_token),
            hicache_storage_pass_prefix_keys,
        )


    def reset(self):
        self._reset_document_prefetch()
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.all_nodes.clear()
        self.key_to_nodes.clear()

    def get_height(self, node):
        return 0

    def clear_storage_backend(self) -> bool:
        if self.enable_storage:
            try:
                # Check if the storage backend has a clear method (for nixl backends)
                if hasattr(self.cache_controller.storage_backend, "clear"):
                    self.cache_controller.storage_backend.clear()
                    logger.info(
                        "Hierarchical cache storage backend cleared successfully!"
                    )
                    return True
                else:
                    logger.warning(
                        f"Storage backend {type(self.cache_controller.storage_backend).__name__} does not support clear operation."
                    )
                    return False
            except Exception as e:
                logger.error(f"Failed to clear hierarchical cache storage backend: {e}")
                return False
        else:
            logger.warning("Hierarchical cache storage backend is not enabled.")
            return False

    def write_backup(self, node: ChunkNode, write_back=False):
        host_indices = self.cache_controller.write(
            device_indices=node.values[0],
            node_id=node.id,
        )
        if host_indices is None:
            self.evict_host(len(node.values[0]))
            host_indices = self.cache_controller.write(
                device_indices=node.values[0],
                node_id=node.id,
            )
        if host_indices is not None:
            node.host_value = host_indices
            assert len(node.host_value) > 0
            self.ongoing_write_through[node.id] = node
        else:
            raise "can't have enough host value!"
            return 0

        return len(host_indices)

    ## fixme: no need to write back to storage
    def write_backup_storage(self, node: ChunkNode):
        return
        prefix_keys = (
            node.get_prefix_hash_values(node.parent)
            if self.hicache_storage_pass_prefix_keys
            else None
        )

        operation_id = self.cache_controller.write_storage(
            node.host_value, node.key, node.hash_value, prefix_keys
        )
        self.ongoing_backup[operation_id] = node
        node.protect_host()

    def _inc_hit_count(self, node: ChunkNode, chunked=False):
        # skip the hit count update for chunked requests
        ""

    def writing_check(self, write_back=False):
        if write_back:
            # blocking till all write back complete
            while len(self.ongoing_write_through) > 0:
                for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
                    finish_event.synchronize()
                    for ack_id in ack_list:
                        del self.ongoing_write_through[ack_id]
                self.cache_controller.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        # NOTE: all ranks has the same ongoing_write_through, can skip sync if empty
        if len(self.ongoing_write_through) == 0:
            return

        finish_count = 0
        for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            # synchronize TP workers to make the same update to radix cache
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )

        finish_count = int(queue_size.item())
        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                backuped_node = self.ongoing_write_through.pop(ack_id)
                # self.dec_lock_ref(backuped_node)
                if self.enable_storage:
                    self.write_backup_storage(backuped_node)
            finish_count -= 1

    def loading_check(self):
        finish_count = 0
        for _, finish_event, ack_list in self.cache_controller.ack_load_queue:
            if not finish_event.query():
                # the KV cache loading is still ongoing
                break
            finish_count += 1
            # no need to sync across TP workers as batch forwarding is synced
            # for ack_id in ack_list:
            #     end_node = self.ongoing_load_back.pop(ack_id)
                # self.dec_lock_ref(end_node)

        # ACK until all events are processed
        del self.cache_controller.ack_load_queue[:finish_count]


    def evictable_size(self):
        return self.evictable_size_

    def evict(self, num_tokens: int):
        print(f"we need to evict {num_tokens} tokens")
        raise f"panic! not enough memories!"
        ""

    def _evict_backuped(self, node: ChunkNode):
        ""

    def evict_host(self, num_tokens: int, exclude: Optional[set[ChunkNode]] = None):
        """
        Evict host cache nodes with host_ref_counter == 0,
        preferring nodes with the lowest hit_count.
        """
        print(
            f"[fusionrag cache] needs {num_tokens} tokens, current available size={self.cache_controller.mem_pool_host.available_size()}")
        tokens_freed = 0

        # Evict eligible nodes in ascending hit-count order.
        evictable_nodes = [
            node for node in self.all_nodes
            if node.host_ref_counter == 0 and node.host_value is not None
            and (node.disk_tensor_path is None or not node.values)
            and (exclude is None or node not in exclude)
        ]
        print(f"evictable_nodes length={len(evictable_nodes)}")
        evictable_nodes.sort(key=lambda node: node.hit_count)
        # for node in self.all_nodes:
        #     print(f"node host_ref_counter {node.host_ref_counter} node text={node.text_without_prefix[:30]}")

        for node in evictable_nodes:
            if tokens_freed >= num_tokens:
                break

            self.cache_controller.mem_pool_host.free(node.host_value)
            tokens_freed += len(node.host_value)

            if node.disk_tensor_path is None:
                del self.key_to_nodes[node.hash_key()]
                self.all_nodes.remove(node)
            else:
                node.host_value = None
        print(f"[fusionrag cache] evict_host {num_tokens} tokens, current nodes = {len(self.all_nodes)} available size={self.cache_controller.mem_pool_host.available_size()}")

        return tokens_freed


    def load_back(
        self, nodes_to_load: List[ChunkNode], hicache: BasePrefixCache, prefix_cache_ids_len: int,
        mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor, list[torch.Tensor]]:
        # todo: more loading policies
        # print(f"nodes_to_load len = {len(nodes_to_load)}")

        if self.document_cache_dir is not None:
            self._restore_document_chunks(nodes_to_load)
        start_time = time.perf_counter()
        if os.environ.get("DEBUG_PRINT", "").lower() in ["true", "1"]:
            for n in nodes_to_load:
                print(f"[load_back] text_without_prefix={n.text_without_prefix[:30]}, host_value={n.host_value}")
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        last_hit_node = nodes_to_load[-1]
        ancester_node = nodes_to_load[0]

        device_indices = self.cache_controller.load(
            host_indices=host_indices, node_id=last_hit_node.id
        )
        ##debug to see if this is working
        # hicache.evict(EvictParams(num_tokens=0))
        if device_indices is None:
            print(f"not enough HBM to load")
            if hicache is not None:
                print(f"try borrow from hicache")
                hicache.evict(EvictParams(num_tokens=len(host_indices)))
                device_indices = self.cache_controller.load(
                    host_indices=host_indices, node_id=last_hit_node.id
                )
        if device_indices is None:
            raise Exception(f"not enough HBM to load")
            # no sufficient GPU memory to load back KV caches
        self.ongoing_load_back[last_hit_node.id] = last_hit_node
        offset = 0
        all_values = []
        for node in nodes_to_load:
            device_hit_index = device_indices[offset : offset + len(node.host_value)]
            node.values.append(device_hit_index)
            all_values.append(
                HitCacheNode(
                    value=device_hit_index,
                    current_position=torch.arange(prefix_cache_ids_len + offset,
                                                  prefix_cache_ids_len + offset + len(node.host_value)).to(
                        device_hit_index.device),
                    original_position=torch.arange(node.cache_prefix_token_len,
                                                   node.cache_prefix_token_len + len(node.host_value)).to(device_hit_index.device),
                )
            )
            offset += len(node.host_value)

        if self.metrics_collector is not None:
            self.metrics_collector.observe_load_back_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_load_back_num_tokens(len(device_indices))

        # return device_indices[:-1], all_values ## left one just for decode
        return device_indices, all_values  ## cache_debug hardcode

    def init_load_back_chunk(self, all_hit_nodes: List[ChunkNode], hicache: BasePrefixCache, prefix_cache_ids_len: int):
        if self.document_cache_dir is not None:
            with torch.profiler.record_function("fusionrag_kv_load"):
                return self.load_back(all_hit_nodes, hicache, prefix_cache_ids_len)
        return self.load_back(all_hit_nodes, hicache, prefix_cache_ids_len)

    def match_prefix(self, params: MatchPrefixParams):
        all_hit_chunk_nodes = []
        host_hit_length = 0
        if params.key.is_kv_gen:
            for node in self.all_nodes:
                if node.is_preprocess_cache == params.key.is_preprocess_kv_gen:
                    if node.is_preprocess_cache:
                        if node.preprocess_cache_key != params.key.preprocess_cache_key:
                            continue
                    prefix_text = params.key.prefix_prompt_text
                    text_without_prefix = params.key.origin_input_text[len(prefix_text):]
                    if text_without_prefix == node.text_without_prefix:
                        node.host_ref_counter += 1
                        if os.environ.get("DEBUG", "0") != "0":
                            print(f"kv gen already run before\ntext={text_without_prefix}\n"
                              f"prefix={prefix_text}\n"
                              f"is_preprocess_cache={node.is_preprocess_cache}")
                        return MatchResult(
                            device_indices=torch.empty(
                                (0,),
                                dtype=torch.int64,
                                device=self.device,
                            ),  ## cache_debug let all be empty on the device.
                            all_hit_chunk_nodes=all_hit_chunk_nodes,
                            host_hit_length=host_hit_length,
                            last_host_node=None,
                            last_device_node=None,
                            no_need_to_run=True,
                        )


        input_text = str(copy.deepcopy(params.key.prefix_prompt_text))
        last_round_found = True
        if len(input_text) == 0:
            print(f"cache_debug fusionrag cache match_prefix skipping prefix.")
        ##
        match_idx = 0
        if params.key.use_preprocess_kv_cache == True:
            print(f"use preprocess cache") ## for debug
        preprocess_cache_key_list = params.key.preprocess_cache_key_list
        cache_is_preprocess_list = params.key.cache_is_preprocess_list
        if cache_is_preprocess_list is None:
            use_preprocess_kv_cache = params.key.use_preprocess_kv_cache
        else:
            use_preprocess_kv_cache = cache_is_preprocess_list[match_idx]
        if use_preprocess_kv_cache or params.key.prefix_prompt_list in [[], None]:
            while len(input_text) > 0 and last_round_found:
                last_round_found = False
                for node in self.all_nodes:
                    if node.is_preprocess_cache == use_preprocess_kv_cache:
                        if use_preprocess_kv_cache:
                            if preprocess_cache_key_list is not None and preprocess_cache_key_list[match_idx] != "skip":
                                if node.preprocess_cache_key != params.key.preprocess_cache_key_list[match_idx]:
                                    continue
                        if len(node.text_without_prefix) > 5 and input_text.startswith(node.text_without_prefix):
                            host_hit_length += node.num_tokens
                            all_hit_chunk_nodes.append(node)
                            node.hit_count += 1
                            if os.environ.get("DEBUG_PRINT", "").lower() in ["true", "1"]:
                                print(f"match_idx={match_idx} load text: {node.text_without_prefix[:10]}......, preprocess={node.is_preprocess_cache}, save_kv_cache={params.key.is_kv_gen}")
                                print(f"text_without_prefix_ids={len(node.text_without_prefix_ids)}")
                                print(f"host_tokens={node.num_tokens}")
                                if node.preprocess_cache_key != "":
                                    print(f"match_idx={match_idx} preprocess_cache_key={node.preprocess_cache_key}")
                                try:
                                    print(f"match_idx={match_idx} prefix_prompt_ids_list={len(params.key.prefix_prompt_ids_list[match_idx])}")
                                except Exception as e:
                                    print(f"match_idx={match_idx}")
                                    print(f"prefix_prompt_ids_list={len(params.key.prefix_prompt_ids_list)}")
                            input_text = input_text[len(node.text_without_prefix) :]
                            last_round_found = True
                            break
                match_idx += 1
        else:
            for prefix_text_ in params.key.prefix_prompt_list:
                text = f"is_preprocess_cache={use_preprocess_kv_cache} text_without_prefix={prefix_text_}"
                hash_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if hash_key in self.key_to_nodes:
                    node = self.key_to_nodes[hash_key]
                    host_hit_length += node.num_tokens
                    all_hit_chunk_nodes.append(node)
                    node.hit_count += 1
                else:
                    break


        return MatchResult(
            device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ), ## cache_debug let all be empty on the device.
            all_hit_chunk_nodes=all_hit_chunk_nodes,
            host_hit_length=host_hit_length,
            last_host_node=None,
            last_device_node=None,
            no_need_to_run=False,
        )

    def insert(
        self,
        key: RadixKey,
        value=None,
        chunked: bool = False,
        priority: int | None = None,
        is_kv_gen: bool = False,
        kv_gen_prefix_len: int = 0,
        is_preprocess_cache: bool = False,
        preprocess_cache_key: str = "",
        text_without_prefix_ids: List[int] = None
    ):
        if is_kv_gen:
            node = ChunkNode()
            node.text_without_prefix = key.origin_input_text[len(key.prefix_prompt_text):]
            if os.environ.get("DEBUG_PRINT", "").lower() in ["true", "1"]:
                print(f"text_without_prefix = {node.text_without_prefix[:30]}")
            node.prefix_text = key.prefix_prompt_text
            node.values = [value]
            node.priority = priority
            node.cache_prefix_token_len = kv_gen_prefix_len
            node.is_preprocess_cache = is_preprocess_cache
            node.preprocess_cache_key = preprocess_cache_key
            node.text_without_prefix_ids = text_without_prefix_ids or []
            target_len = len(node.text_without_prefix)
            pos = bisect.bisect_left(
                self.all_nodes,
                -target_len,
                key=lambda n: -len(n.text_without_prefix)
            )
            self.all_nodes.insert(pos, node)
            self.write_backup(node)
            self.key_to_nodes[node.hash_key()] = node
            node.values = []
            return node


    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: Any,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        ""

    def terminate_prefetch(self, req_id: str):
        ""

    def check_prefetch_progress(self, req_id: str) -> bool:
        ""

    def can_terminate_prefetch(self, operation: PrefetchOperation):
        ""

    def drain_storage_control_queues(self):
        """
        Combine prefetch revoke, backup ack, and host mem release checks
        to minimize TP synchronization and Python overhead.
        """

    def check_hicache_events(self):
        self.writing_check()
        self.loading_check()
        if self.enable_storage:
            self.drain_storage_control_queues()
        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_storage_metrics(
                self.cache_controller.storage_backend.get_stats()
            )

    def ready_to_load_host_cache(self) -> int:
        """
        Notify the cache controller to start the KV cache loading.
        Return the consumer index for the schedule batch manager to track.
        """
        profile = ttft_probe.output is not None
        if profile:
            ttft_probe.mark("origin")
        return self.cache_controller.start_loading(profile=profile)

    def cache_unfinished_req(self, req: Req, chunked=False):
        print(f"cache_unfinished_req")
        ""

    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None:
        if req.is_kv_gen and req.save_preprocess_cache:
            print(f"save preprocess cache") ## for debug
        if os.environ.get("DEBUG_PRINT", "").lower() in ["true", "1"]:
            logger.error(
                f"fusionrag cache_finished_req: rid={req.rid} "
                f"is_kv_gen={req.is_kv_gen} save_raw_cache={req.save_raw_cache} "
                f"save_preprocess_cache={req.save_preprocess_cache} no_need_to_run={req.no_need_to_run} "
                f"preprocess_cache_key={req.preprocess_cache_key}"
            )
        if req.no_need_to_run:
            return
        if req.is_kv_gen:
            token_ids = req.origin_input_ids
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(token_ids)
            ]
            radix_key = RadixKey(
                token_ids=token_ids,
                origin_input_text=req.origin_input_text,
                prefix_prompt_text=req.prefix_prompt,
            )
            values = kv_indices.to(dtype=torch.int64, copy=True)

            kv_prefix_len = req.kv_gen_prefix_len ## this is right
            text_without_prefix_ids = req.origin_input_ids[kv_prefix_len:]
            assert len(text_without_prefix_ids) == len(values) - kv_prefix_len
            if os.environ.get("DEBUG_PRINT", "").lower() in ["true", "1"]:
                print(f"saving to cache, text_without_prefix_ids={len(text_without_prefix_ids)}\n"
                      f"values={len(values)}\n"
                      f"kv_prefix_len={kv_prefix_len}\n")
            node = self.insert(
                radix_key,
                values[kv_prefix_len:],
                priority=0,
                is_kv_gen=True,
                kv_gen_prefix_len=kv_prefix_len,
                is_preprocess_cache=req.save_preprocess_cache,
                preprocess_cache_key=req.preprocess_cache_key,
                text_without_prefix_ids=text_without_prefix_ids
            )
            # Keep source GPU slots allocated until the asynchronous host copy completes.
            self.writing_check(write_back=True)

            if self.document_cache_dir is not None and req.save_raw_cache:
                self._persist_document(node)
                if self.replicated_document_kv:
                    self.token_to_kv_pool_host.free(node.host_value)
                    node.host_value = None

            kv_committed_len = req.pop_committed_kv_cache()
            token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(token_ids)
            ]
            if req.kv_gen_use_radix_prefix:
                self.cache_controller.mem_pool_device_allocator.free(kv_indices[kv_prefix_len:])
            else:
                self.cache_controller.mem_pool_device_allocator.free(kv_indices)

        for node_idx, node in enumerate(req.hit_chunk_nodes):
            value_to_remove = req.hit_chunk_values[node_idx].value
            if not req.is_kv_gen:
                if node.host_ref_counter > 0:
                    node.host_ref_counter -= 1
            try:
                for value_idx, value in enumerate(node.values):
                    if torch.equal(value, value_to_remove):
                        del node.values[value_idx]
                        break
            except Exception as E:
                print(f"cache finish req error. E={E}, node={node}")

        if req.clear_cache:
            self.evict_host(200000) ## cache_debug: just give a big number.

        # self.req_to_token_pool.free(req) ## this fill be freed in release_kv_cache(

    def dec_lock_ref(self, node: ChunkNode):
        ""

    def inc_lock_ref(self, node: ChunkNode):
        ""
