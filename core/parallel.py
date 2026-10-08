"""Run independent per-frame work on several CPU cores, keeping frame order."""
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor


def default_workers():
    return os.cpu_count() or 1


def ordered_map(tasks, workers=None):
    """Yield (context, result) for each (function, args, context) task, in order.

    A task whose function is None yields result None without using a worker.
    At most a few tasks per worker are in flight, which bounds memory for long
    videos. With one worker everything runs in this process.
    """
    workers = default_workers() if workers is None else workers
    if workers <= 1:
        for function, args, context in tasks:
            yield context, None if function is None else function(*args)
        return

    window = 2 * workers
    pending = deque()
    executor = ProcessPoolExecutor(max_workers=workers)
    try:
        for function, args, context in tasks:
            pending.append((None if function is None else executor.submit(function, *args), context))
            while len(pending) > window or (pending and (pending[0][0] is None or pending[0][0].done())):
                future, context = pending.popleft()
                yield context, None if future is None else future.result()
        while pending:
            future, context = pending.popleft()
            yield context, None if future is None else future.result()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
