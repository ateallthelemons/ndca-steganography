"""File mapping and explicitly selected embedding profiles."""
from dataclasses import dataclass
from math import isfinite

BLOCK_SIZE = 16
MAGIC = b'NDV5'
EMBEDDING_ID = 14
HEADER_COPIES = 3


@dataclass(frozen=True)
class Settings:
    """Immutable numerical settings passed explicitly through the pipeline."""
    width: int
    delta: int
    cost: str

    def __post_init__(self):
        if (type(self.width) is not int or self.width not in (2, 3)
                or not isfinite(self.delta) or self.delta <= 0
                or self.cost not in ('crf', 'cqp')):
            raise ValueError('Invalid embedding settings')


PROFILES = {
    'base': Settings(width=2, delta=80, cost='crf'),
    'robust': Settings(width=2, delta=120, cost='crf'),
}


def add_profile_argument(parser):
    parser.add_argument('--profile', required=True, choices=tuple(PROFILES),
                        help='Embedding profile; use the same profile to encode and decode')
