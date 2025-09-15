import torch
import torch.distributed as dist


def _maybe_reshape_or_expand(x, target_rank, last_dim=None):
    """
    If target_rank == x.dim(): return x
    If target_rank == x.dim()+1: unsqueeze(-1) and expand last_dim.
    Otherwise, fallback to view/reshape (may fail if sizes mismatch).
    """
    if target_rank == x.dim():
        return x
    if target_rank == x.dim() + 1:
        x2 = x.unsqueeze(-1)
        if last_dim is None:  # cannot infer; keep as unsqueezed singleton
            return x2
        shape = list(x2.shape)
        shape[-1] = last_dim
        return x2.expand(*shape)
    # Fallback: trust reshape with -1 placeholders
    return x.reshape(*([-1] * target_rank))


def _all_gather_cat(x, dim):
    if not dist.is_available() or not dist.is_initialized():
        return x
    ws = dist.get_world_size()
    tensors = [torch.empty_like(x) for _ in range(ws)]
    dist.all_gather(tensors, x)
    return torch.cat(tensors, dim=dim)


def _all_reduce_sum(x):
    if not dist.is_available() or not dist.is_initialized():
        return x
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def _reduce_scatter_sum_cat(x, dim):
    if not dist.is_available() or not dist.is_initialized():
        return x
    # Split equally along dim and reduce_scatter. Requires equal chunks.
    ws = dist.get_world_size()
    chunks = list(torch.chunk(x, ws, dim=dim))
    out = torch.empty_like(chunks[0])
    dist.reduce_scatter_tensor(out, torch.cat(chunks, dim=0), op=dist.ReduceOp.SUM)
    return out


def _shard_take(x, dim):
    """Return the local shard of x along dim by world rank.

    Splits equally into world_size chunks and returns chunk[rank].
    If dist is not initialized, returns x unchanged.
    """
    if not dist.is_available() or not dist.is_initialized():
        return x
    ws = dist.get_world_size()
    rank = dist.get_rank()
    chunks = list(torch.chunk(x, ws, dim=dim))
    return chunks[rank]
