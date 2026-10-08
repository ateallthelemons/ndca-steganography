"""Occasional, flushed progress lines for long video operations."""
import sys
from time import monotonic


class Progress:
    def __init__(self, label, *, enabled=True):
        self.label = label
        self.enabled = enabled
        self.started = self.last_report = monotonic()
        if enabled:
            print(f'{label}: starting...', file=sys.stderr, flush=True)

    def update(self, frames, detail, *, final=False):
        if not self.enabled:
            return
        now = monotonic()
        if not final and now - self.last_report < 5:
            return
        self.last_report = now
        status = 'complete; ' if final else ''
        print(f'{self.label}: {status}{frames} frames, {detail}, '
              f'{now - self.started:.0f}s elapsed', file=sys.stderr, flush=True)
