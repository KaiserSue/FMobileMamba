"""Per-rank NVML sampling and fold-local GPU reports."""
import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class MonitorConfig:
    enabled: bool = True
    sample_interval_seconds: float = 0.5
    report_interval_seconds: float = 30.0

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('gpu_monitor.enabled must be boolean')
        for value in (self.sample_interval_seconds, self.report_interval_seconds):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('gpu_monitor intervals must be positive finite numbers')
        if self.report_interval_seconds <= self.sample_interval_seconds:
            raise ValueError('gpu_monitor report interval must exceed sample interval')


@dataclass(frozen=True)
class DeviceIdentity:
    rank: int
    local_rank: int
    device_type: str
    cuda_visible_index: int
    gpu_name: str
    gpu_uuid: str = None
    pci_bus_id: str = None


@dataclass(frozen=True)
class RankEpochStats:
    epoch: int
    rank: int
    local_rank: int
    gpu_id: str
    status: str
    error: str
    sample_count: int
    util_avg_pct: float
    util_max_pct: float
    device_memory_used_mib: float
    process_peak_mib: float


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _snapshot_path(logdir, rank):
    return Path(logdir) / '.gpu_monitor_rank_{}.json'.format(rank)


def append_record(logdir, record):
    with (Path(logdir) / 'GPU_useage.txt').open('a', encoding='utf-8') as stream:
        stream.write(record + '\n')
        stream.flush()


def format_device_group(identities):
    rows = ['[GPU monitor {}] device confirmation; CUDA indices are process-visible, not physical GPU indices'.format(_now())]
    for item in sorted(identities, key=lambda entry: entry.rank):
        rows.append('rank={} local_rank={} device={} visible_cuda={} name={} UUID={} PCI={}'.format(
            item.rank, item.local_rank, item.device_type, item.cuda_visible_index,
            item.gpu_name, item.gpu_uuid or 'unavailable', item.pci_bus_id or 'unavailable'))
    return '\n'.join(rows)


def _format_rank(item):
    def value(number, unit=''):
        return 'unavailable' if number is None else '{:.1f}{}'.format(number, unit)
    return ('rank={rank} gpu={gpu_id} status={status} samples={sample_count} '
            'avg={avg} max={maximum} device_used={memory} process_peak={peak} error={error}').format(
        rank=item['rank'], gpu_id=item['gpu_id'], status=item['status'],
        sample_count=item['sample_count'], avg=value(item['util_avg_pct'], '%'),
        maximum=value(item['util_max_pct'], '%'),
        memory=value(item['device_memory_used_mib'], ' MiB'),
        peak=value(item['process_peak_mib'], ' MiB'), error=item['error'] or '-')


def format_epoch_group(epoch_number, rank_stats):
    rows = ['[GPU monitor {}] epoch={} complete; device utilization/memory are device-wide; process peak is PyTorch allocated memory'.format(_now(), epoch_number)]
    rows.extend(_format_rank(asdict(item)) for item in sorted(rank_stats, key=lambda item: item.rank))
    return '\n'.join(rows)


def read_periodic_group(epoch_number, now, world_size, logdir, window_seconds):
    ranks = []
    for rank in range(world_size):
        path = _snapshot_path(logdir, rank)
        if not path.exists():
            ranks.append(dict(rank=rank, gpu_id='unavailable', status='unavailable', error='snapshot missing',
                              sample_count=0, util_avg_pct=None, util_max_pct=None,
                              device_memory_used_mib=None, process_peak_mib=None))
            continue
        item = json.loads(path.read_text(encoding='utf-8'))
        timestamp = datetime.fromisoformat(item['timestamp'])
        if item['epoch'] != epoch_number or (now - timestamp).total_seconds() > 2 * window_seconds:
            item.update(status='stale', error='snapshot expired', sample_count=0,
                        util_avg_pct=None, util_max_pct=None, device_memory_used_mib=None)
        if item['status'] != 'stale':
            item['sample_count'] = item['window_count']
            item['util_avg_pct'] = (item['window_total'] / item['window_count']
                                    if item['window_count'] else None)
            item['util_max_pct'] = item['window_maximum']
            item['device_memory_used_mib'] = item['window_memory']
        ranks.append(item)
    return dict(timestamp=now.isoformat(), epoch=epoch_number, window_seconds=window_seconds, ranks=ranks)


def format_periodic_group(record):
    rows = ['[GPU monitor {}] epoch={} window={:.1f}s; device utilization/memory are device-wide; process peak is PyTorch allocated memory'.format(
        record['timestamp'], record['epoch'], record['window_seconds'])]
    rows.extend(_format_rank(item) for item in sorted(record['ranks'], key=lambda item: item['rank']))
    return '\n'.join(rows)


def _nvml_device(cuda_index):
    import torch
    import pynvml
    props = torch.cuda.get_device_properties(cuda_index)
    uuid = getattr(props, 'uuid', None)
    pci = getattr(props, 'pci_bus_id', None)
    if not pci:
        import ctypes
        buffer = ctypes.create_string_buffer(32)
        result = torch.cuda.cudart().cudaDeviceGetPCIBusId(buffer, len(buffer), cuda_index)
        if result == 0:
            pci = buffer.value.decode()
    if not uuid and not pci:
        raise RuntimeError('PyTorch exposes no CUDA UUID or PCI bus ID for safe NVML mapping')
    matches = []
    for index in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        nv_uuid = pynvml.nvmlDeviceGetUUID(handle)
        nv_pci = pynvml.nvmlDeviceGetPciInfo(handle).busId
        if isinstance(nv_uuid, bytes):
            nv_uuid = nv_uuid.decode()
        if isinstance(nv_pci, bytes):
            nv_pci = nv_pci.decode()
        if (uuid and str(uuid) == nv_uuid) or (pci and str(pci).lower() == nv_pci.lower()):
            matches.append((handle, nv_uuid, nv_pci))
    if len(matches) != 1:
        raise RuntimeError('CUDA to NVML device mapping is missing or ambiguous')
    return matches[0]


def describe_device(rank, local_rank, model_parameter_device):
    import torch
    device = torch.device(model_parameter_device)
    if device.type != 'cuda':
        return DeviceIdentity(rank, local_rank, device.type, -1, 'unavailable')
    index = device.index if device.index is not None else torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    uuid = pci = None
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            _, uuid, pci = _nvml_device(index)
        finally:
            pynvml.nvmlShutdown()
    except (ImportError, RuntimeError, OSError, AttributeError):
        pass  # Optional identifiers never hide the PyTorch-confirmed device.
    return DeviceIdentity(rank, local_rank, device.type, index, name, uuid, pci)


class GpuMonitor:
    def __init__(self, local_rank, rank, config, logdir):
        self.local_rank, self.rank, self.config = local_rank, rank, config
        self.logdir = Path(logdir)
        self.gpu_id = 'cuda:{}'.format(local_rank)
        self.status, self.error = 'disabled', 'monitor disabled'
        self.nvml = self.handle = None
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.epoch = None
        if not config.enabled:
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle, uuid, _ = _nvml_device(local_rank)
            self.gpu_id = '{} (visible cuda:{})'.format(uuid, local_rank)
            self.status, self.error = 'ok', None
        except (ImportError, RuntimeError, OSError, AttributeError) as error:
            self.status, self.error = 'unavailable', str(error)
            if self.nvml is not None:
                self.nvml.nvmlShutdown()
                self.nvml = None

    def start_epoch(self, epoch_number):
        if epoch_number <= 0:
            raise ValueError('epoch_number must be positive')
        self.epoch = epoch_number
        self.count = self.total = 0
        self.window_count = self.window_total = 0
        self.window_maximum = self.window_memory = None
        self.window_start = time.monotonic()
        self.maximum = self.memory = None
        self.iteration = 0
        self.peak = None
        self.progress_time = None
        self.stop_event.clear()
        self._publish()
        if self.status == 'ok':
            self.thread = threading.Thread(target=self._sample_loop, daemon=True)
            self.thread.start()

    def _sample_loop(self):
        while not self.stop_event.wait(self.config.sample_interval_seconds):
            with self.lock:
                if time.monotonic() - self.window_start >= self.config.report_interval_seconds:
                    self.window_count = self.window_total = 0
                    self.window_maximum = self.window_memory = None
                    self.window_start = time.monotonic()
                try:
                    usage = self.nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                    memory = self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used / 1048576
                    self.count += 1
                    self.total += usage
                    self.window_count += 1
                    self.window_total += usage
                    self.window_maximum = usage if self.window_maximum is None else max(self.window_maximum, usage)
                    self.window_memory = memory
                    self.maximum = usage if self.maximum is None else max(self.maximum, usage)
                    self.memory = memory
                except self.nvml.NVMLError as error:
                    self.status, self.error = 'degraded', str(error)
                self._publish()

    def _stats(self):
        return dict(epoch=self.epoch, rank=self.rank, local_rank=self.local_rank,
                    gpu_id=self.gpu_id, status=self.status, error=self.error,
                    sample_count=self.count,
                    util_avg_pct=self.total / self.count if self.count else None,
                    util_max_pct=self.maximum, device_memory_used_mib=self.memory,
                    process_peak_mib=self.peak, iteration=self.iteration,
                    process_peak_updated_at=self.progress_time, timestamp=_now(),
                    window_count=self.window_count, window_total=self.window_total,
                    window_maximum=self.window_maximum, window_memory=self.window_memory)

    def _publish(self):
        path = _snapshot_path(self.logdir, self.rank)
        temp = path.with_name(path.name + '.tmp')
        try:
            temp.write_text(json.dumps(self._stats()), encoding='utf-8')
            os.replace(temp, path)
        except OSError as error:
            self.status, self.error = 'degraded', 'snapshot write failed: {}'.format(error)

    def publish_progress(self, iteration, process_peak_bytes):
        if iteration < 0 or process_peak_bytes < 0:
            raise ValueError('progress and memory must be nonnegative')
        with self.lock:
            self.iteration = iteration
            self.peak = process_peak_bytes / 1048576
            self.progress_time = _now()
            self._publish()

    def stop_epoch(self, process_peak_bytes):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        self.publish_progress(self.iteration, process_peak_bytes)
        values = self._stats()
        return RankEpochStats(*(values[key] for key in RankEpochStats.__dataclass_fields__))

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        _snapshot_path(self.logdir, self.rank).unlink(missing_ok=True)
        if self.nvml is not None:
            self.nvml.nvmlShutdown()
            self.nvml = None


def create_monitor(local_rank, rank, config, logdir):
    return GpuMonitor(local_rank, rank, config, logdir)
