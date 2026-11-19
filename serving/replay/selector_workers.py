"""Per-process selector affinity and reusable offline draft keys."""

import asyncio
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import json
import os
from pathlib import Path
import time


_DRAFT = None
_SCORE_LAYERS = None
_DIAG = None


def cpu_snapshot():
    values = {}
    with open("/proc/stat") as f:
        for line in f:
            label, *fields = line.split()
            if label.startswith("cpu") and label[3:].isdigit():
                counts = [int(field) for field in fields[:8]]
                values[int(label[3:])] = sum(counts), counts[3] + counts[4]
    return values


def pick_cpus(cores_per_socket, sample_s=2.0):
    allowed = os.sched_getaffinity(0)
    topology = {}
    for cpu in allowed:
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        socket = int((base / "topology/physical_package_id").read_text())
        core = int((base / "topology/core_id").read_text())
        topology.setdefault(socket, {}).setdefault(core, []).append(cpu)
    start = cpu_snapshot()
    time.sleep(sample_s)
    end = cpu_snapshot()
    load = {}
    for cpu in allowed:
        total = end[cpu][0] - start[cpu][0]
        load[cpu] = 1 - (end[cpu][1] - start[cpu][1]) / max(total, 1)
    sockets = []
    measurements = []
    for pairs in topology.values():
        ranked = sorted(pairs.values(), key=lambda pair: (sum(load[c] for c in pair), min(pair)))
        keep = ranked[:cores_per_socket]
        sockets.append([min(pair, key=lambda cpu: (load[cpu], cpu)) for pair in keep])
        measurements.append([{"siblings": sorted(pair), "active": [round(load[c], 4) for c in sorted(pair)]}
                             for pair in ranked])
    return sockets, measurements


def placement(sockets, workers, threads):
    groups = []
    for i in range(workers):
        socket = sockets[i % len(sockets)]
        offset = (i // len(sockets)) * threads
        cpus = socket[offset:offset + threads]
        if len(cpus) != threads:
            raise ValueError("Insufficient physical cores for the requested placement")
        groups.append(cpus)
    return groups


def init_torch(threads, cpus):
    if cpus:
        os.sched_setaffinity(0, cpus)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    return torch


def precompute_part(data, keys, path, threads, cpus):
    torch = init_torch(threads, cpus)
    from selector import Selector
    worker = Selector({key: data[key] for key in keys})
    worker.prepare()
    temp = path.with_suffix(".part")
    torch.save(worker.keys, temp)
    temp.replace(path)
    return len(worker.keys)


def precompute(data, directory, jobs=8):
    directory.mkdir(exist_ok=True)
    keys = list(data["blocks"])
    groups, _ = pick_cpus(16)
    seats = placement(groups, jobs, 2)
    started = time.monotonic()
    with ProcessPoolExecutor(jobs, mp_context=get_context("spawn")) as pool:
        pending = []
        for i, cpus in enumerate(seats):
            selected = keys[i::jobs]
            part = {key: {"draft_ids": data["blocks"][key]["draft_ids"]} for key in selected}
            path = directory / f"keys-{i:02d}.pt"
            pending.append(pool.submit(precompute_part, part, selected, path, len(cpus), cpus))
        count = sum(job.result() for job in pending)
    assert count == len(keys)
    result = {"blocks": count, "workers": jobs, "cores": seats, "seconds": time.monotonic() - started,
              "bytes": sum(path.stat().st_size for path in directory.glob("keys-*.pt"))}
    print(f"Offline CPU draft keys {count} in {result['seconds']:.1f} s", flush=True)
    return result


def init_selector(workload, key_files, cpus, threads, device, score_layers):
    global _DRAFT, _SCORE_LAYERS, _DIAG
    torch = init_torch(threads, cpus)
    from selector import Selector
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    with open(workload) as f:
        source = json.load(f)
    bank = {key: {field: block[field] for field in ("draft_offsets", "target_offsets")}
            for key, block in source["blocks"].items()}
    _DRAFT = Selector(bank, device=device)
    for path in key_files:
        _DRAFT.keys.update(torch.load(path, mmap=True, map_location="cpu", weights_only=True))
    assert _DRAFT.keys.keys() == bank.keys()
    _SCORE_LAYERS = score_layers
    _DIAG = {"pid": os.getpid(), "cpus": sorted(os.sched_getaffinity(0)), "threads": threads, "device": device}
    if device.startswith("cuda"):
        _DIAG["gpu_allocated_mib"] = torch.cuda.memory_allocated(device) / 1024**2
    return _DIAG


def select_one(row, reference=False):
    from hashlib import sha256
    from resource import RUSAGE_SELF, getrusage
    from struct import pack
    import torch
    started = time.perf_counter()
    cpu_started = time.process_time()
    counts = getrusage(RUSAGE_SELF)
    size = len(row["draft_query"])
    if not row["blocks"]:
        raise ValueError("A selector job needs document keys")
    if reference or not _SCORE_LAYERS:
        mask, detail = _DRAFT.select(row, .3, profile=_DRAFT.device.type == "cuda", reference=reference)
    else:
        mask, detail = _DRAFT.select(row, .3, profile=True, score_layers=_SCORE_LAYERS)
    cpu_time = (time.process_time() - cpu_started) * 1000
    after = getrusage(RUSAGE_SELF)
    detail.update(selector_cpu_ms=cpu_time, selector_system_ms=(after.ru_stime-counts.ru_stime)*1000,
                  selector_minor_faults=after.ru_minflt-counts.ru_minflt,
                  selector_major_faults=after.ru_majflt-counts.ru_majflt,
                  selector_involuntary_switches=after.ru_nivcsw-counts.ru_nivcsw,
                  selector_query_tokens=size, selector_history_tokens=sum(_DRAFT.keys[k].shape[-2] for k in row["blocks"]),
                  mask_sha256=sha256(pack(f"<{len(mask)}I", *mask)).hexdigest())
    if _DRAFT.device.type == "cuda":
        detail["selector_cuda_peak_mib"] = torch.cuda.max_memory_reserved(_DRAFT.device) / 1024**2
        detail["selector_cuda_rank"] = _DRAFT.device.index
    return started, time.perf_counter(), mask, detail


def warm_worker(rows):
    for row in rows:
        select_one(row)
    return _DIAG


class WorkerPool:
    def __init__(self, cpus, threads, workload, directory, devices=None, score_layers=None):
        devices = devices or ["cpu"] * len(cpus)
        assert len(cpus) == len(devices)
        self.count = len(cpus)
        self.configuration = dict(workers=self.count, cores=cpus, threads=threads, devices=devices,
                                  score_layers=score_layers)
        key_files = sorted(directory.glob("keys-*.pt"))
        assert key_files
        self.pools = []
        try:
            for group, device in zip(cpus, devices):
                pool = ProcessPoolExecutor(1, mp_context=get_context("spawn"), initializer=init_selector,
                                           initargs=(workload, key_files, group, threads, device, score_layers))
                self.pools.append(pool)
        except BaseException:
            self.close()
            raise
        self.free = None

    def warm(self, rows):
        tasks = [pool.submit(warm_worker, rows) for pool in self.pools]
        self.configuration["observed"] = [task.result() for task in tasks]
        return self.configuration

    def reset_queue(self):
        self.free = asyncio.Queue(maxsize=self.count)
        for pool in self.pools:
            self.free.put_nowait(pool)

    async def select(self, row, reference=False):
        pool = await self.free.get()
        try:
            payload = {field: row[field] for field in ("qid", "draft_query", "blocks", "head_len")}
            future = pool.submit(select_one, payload, reference)
            return await asyncio.wrap_future(future)
        finally:
            self.free.put_nowait(pool)

    def close(self):
        for pool in self.pools:
            pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
