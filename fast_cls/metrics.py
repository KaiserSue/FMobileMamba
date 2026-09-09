"""Detached device accumulation with a single collective per flush."""
import torch
import torch.distributed as dist


class DeviceMetrics:
    def __init__(self, names, device):
        if not names or len(set(names)) != len(names):
            raise ValueError('Metric names must be nonempty and unique')
        self.names = tuple(names)
        self.sums = torch.zeros(len(names), dtype=torch.float64, device=device)
        self.counts = torch.zeros_like(self.sums)

    @torch.no_grad()
    def update(self, sums, counts):
        if set(sums) != set(self.names) or set(counts) != set(self.names):
            raise ValueError('Metric keys must match names')
        for i, name in enumerate(self.names):
            for value in (sums[name], counts[name]):
                if not isinstance(value, torch.Tensor) or value.ndim != 0:
                    raise ValueError('Metric contributions must be scalar tensors')
            self.sums[i].add_(sums[name].detach().to(self.sums))
            self.counts[i].add_(counts[name].detach().to(self.counts))

    @torch.no_grad()
    def flush(self):
        packed = torch.stack((self.sums, self.counts))
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        sums, counts = packed.cpu().tolist()
        self.sums.zero_()
        self.counts.zero_()
        return {name: {'sum': total, 'count': count, 'mean': total / count}
                for name, total, count in zip(self.names, sums, counts) if count > 0}
