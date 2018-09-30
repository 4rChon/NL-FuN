import scipy.signal
import numpy as np
from collections import namedtuple


def get_network(NetworkClass, env, config):
        network = NetworkClass(env.observation_spec(), config)
        return network


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def get_device_ix(task, device, gpus=2, cpus=8):
    if device == 'gpu':
        return task % gpus
    elif device == 'cpu':
        return task % cpus
    return 0


def pad_batches(batch_1, batch_2, prev_batch_1, prev_batch_2):
    batch_1_len = len(batch_1.s_diff)
    batch_2_len = len(batch_2.s_diff)

    if batch_1_len > batch_2_len:
        if prev_batch_1 is not None:
            batch_2 = pad_with_batch(batch_2, batch_1_len, prev_batch_2)
        else:
            batch_2 = pad(batch_2, batch_1_len)
    if batch_2_len > batch_1_len:
        if prev_batch_1 is not None:
            batch_1 = pad_with_batch(batch_1, batch_2_len, prev_batch_1)
        else:
            batch_1 = pad(batch_1, batch_2_len)
    return batch_1, batch_2


def truncate_batches(batch_1, batch_2):
    batch_1_len = len(batch_1.s_diff)
    batch_2_len = len(batch_2.s_diff)

    if batch_1_len < batch_2_len:
        batch_2 = truncate(batch_2, batch_1_len)
    if batch_2_len < batch_1_len:
        batch_1 = truncate(batch_1, batch_2_len)
    return batch_1, batch_2


Batch = namedtuple("Batch", ["obs", "actions", "manager_returns", "worker_returns", "s_diff", "ri", "g_in", "gsum",
                             "g_prev", "idx", "features"])


def pad_with_batch(batch, pad_to, prev_batch):
    pad_to = pad_to - len(batch.s_diff)
    manager_returns = list(batch.manager_returns)
    worker_returns = list(batch.worker_returns)
    s_diff = list(batch.s_diff)
    ri = list(batch.ri)
    g_in = list(batch.g_in)
    gsum = list(batch.gsum)
    g_prev = list(batch.g_prev)

    manager_returns[:0] = prev_batch.manager_returns[-pad_to:]
    worker_returns[:0] = prev_batch.worker_returns[-pad_to:]
    s_diff[:0] = prev_batch.s_diff[-pad_to:]
    ri[:0] = prev_batch.ri[-pad_to:]
    g_in[:0] = prev_batch.g_in[-pad_to:]
    gsum[:0] = prev_batch.gsum[-pad_to:]
    g_prev[:0] = prev_batch.g_prev[-pad_to:]

    return Batch(batch.obs, batch.actions, manager_returns, worker_returns, s_diff, ri, g_in, gsum,
                 g_prev, batch.idx, batch.features)


def pad(batch, pad_to):
    pad_to = (pad_to - len(batch.s_diff), 0)
    manager_returns = list(batch.manager_returns)
    worker_returns = list(batch.worker_returns)
    s_diff = list(batch.s_diff)
    ri = list(batch.ri)
    g_in = list(batch.g_in)
    gsum = list(batch.gsum)
    g_prev = list(batch.g_prev)

    manager_returns = np.pad(manager_returns, pad_to, 'edge')
    worker_returns = np.pad(worker_returns, pad_to, 'edge')

    s_diff[:0] = [np.copy(s_diff[0]) for _ in range(pad_to[0])]
    ri = np.pad(ri, pad_to, 'edge')

    g_in[:0] = [np.copy(g_in[0]) for _ in range(pad_to[0])]
    gsum[:0] = [np.copy(gsum[0]) for _ in range(pad_to[0])]
    g_prev[:0] = [np.copy(g_prev[0]) for _ in range(pad_to[0])]

    return Batch(batch.obs, batch.actions, manager_returns, worker_returns, s_diff, ri, g_in, gsum,
                 g_prev, batch.idx, batch.features)


def truncate(batch, truncate_to):
    truncate_to = len(batch.s_diff) - truncate_to
    manager_returns = list(batch.manager_returns)
    worker_returns = list(batch.worker_returns)
    s_diff = list(batch.s_diff)
    ri = list(batch.ri)
    g_in = list(batch.g_in)
    gsum = list(batch.gsum)
    g_prev = list(batch.g_prev)

    manager_returns = manager_returns[truncate_to:]
    worker_returns = worker_returns[truncate_to:]
    s_diff = s_diff[truncate_to:]
    ri = ri[truncate_to:]
    g_in = g_in[truncate_to:]
    gsum = gsum[truncate_to:]
    g_prev = g_prev[truncate_to:]

    return Batch(batch.obs, batch.actions, manager_returns, worker_returns, s_diff, ri, g_in, gsum,
                 g_prev, batch.idx, batch.features)
