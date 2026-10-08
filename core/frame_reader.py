"""Detect the fixed NDV4/legacy and NDV5 bootstrap mappings from video pixels."""
from .framing import MessageReader
from .profiles import ROBUST80


class FrameReader:
    def __init__(self):
        self.candidates = [MessageReader(), MessageReader(ROBUST80)]
        self.selected = None
        self.errors = []

    @property
    def received(self):
        if self.selected is not None:
            return self.selected.received
        return max((r.received for r in self.candidates), key=len, default=b'')

    def _read(self, operation):
        if self.selected is not None:
            return operation(self.selected)
        for reader in tuple(self.candidates):
            try:
                result = operation(reader)
            except ValueError as error:
                # Once metadata validates, a body failure must stay a failure.
                if reader.header is not None:
                    raise
                self.errors.append(f'{reader.bootstrap_profile.version}: {error}')
                self.candidates.remove(reader)
                continue
            if reader.header is not None or result is not None:
                self.selected = reader
                self.candidates = [reader]
                return result
        if not self.candidates:
            raise ValueError('Cannot recover the video bootstrap. ' + '; '.join(self.errors))
        return None

    def feed_frame(self, plane):
        return self._read(lambda reader: reader.feed_frame(plane))

    def finish(self):
        return self._read(lambda reader: reader.feed(b'', final=True))
