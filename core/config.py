"""Current file mapping and reusable numerical settings."""
from dataclasses import dataclass
from math import isfinite

BLOCK_SIZE = 16
MAGIC = b'NDV5'
EMBEDDING_ID = 14
HEADER_COPIES = 3


@dataclass(frozen=True)
class Settings:
    """Numerical options for local experiments; file tools use CURRENT only."""
    width: int = 2
    delta: int = 80
    cost: str = 'crf'

    def __post_init__(self):
        if (type(self.width) is not int or self.width not in (2, 3)
                or not isfinite(self.delta) or self.delta <= 0
                or self.cost not in ('crf', 'cqp')):
            raise ValueError('Invalid embedding settings')


CURRENT = Settings()
