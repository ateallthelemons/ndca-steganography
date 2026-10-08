"""Immutable NDV4 body profiles. IDs are wire-format constants, never reassigned."""
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class Profile:
    id: int
    name: str
    stc: str
    width: int
    delta: int
    cost: str
    height: int


LEGACY = Profile(1, 'legacy', 'legacy', 3, 40, 'crf', 2)
# Assignment order is documented in docs/NDV4.md and frozen by regression tests.
_profiles = {1: LEGACY}
for width in (2, 3):
    for delta in (30, 40, 50):
        for cost in ('crf', 'cqp'):
            ident = len(_profiles) + 1
            _profiles[ident] = Profile(ident, f'stc3-r{width}-d{delta}-{cost}',
                                       'matrix', width, delta, cost, 3)
PROFILES = MappingProxyType(_profiles)


def get_profile(value=LEGACY):
    if isinstance(value, Profile):
        if PROFILES.get(value.id) == value:
            return value
    elif type(value) is int:
        if value in PROFILES:
            return PROFILES[value]
    elif isinstance(value, str):
        for profile in PROFILES.values():
            if value == profile.name:
                return profile
    raise ValueError(f'Unsupported embedding profile: {value}')


def select_profile(profile=None, stc=None, rate=None, delta=None, cost=None):
    settings = (stc, rate, delta, cost)
    if profile is not None:
        if any(value is not None for value in settings):
            raise ValueError('Use --profile or individual coding settings, not both')
        return get_profile(profile)
    if all(value is None for value in settings):
        return LEGACY
    stc = stc or 'matrix'
    rate = rate or ('1/3' if stc == 'legacy' else '1/2')
    delta = 40 if delta is None else delta
    cost = cost or 'crf'
    for candidate in PROFILES.values():
        if (candidate.stc, f'1/{candidate.width}', candidate.delta, candidate.cost) == (stc, rate, delta, cost):
            return candidate
    raise ValueError('Unsupported coding settings; legacy requires rate 1/3, delta 40, CRF costs')


def add_profile_arguments(parser):
    parser.add_argument('--profile', choices=[p.name for p in PROFILES.values()])
    parser.add_argument('--stc', choices=('legacy', 'matrix'))
    parser.add_argument('--rate', choices=('1/3', '1/2'), help='STC message bits per carrier, before RS overhead')
    parser.add_argument('--delta', type=int, choices=(30, 40, 50))
    parser.add_argument('--cost', choices=('crf', 'cqp'), help='Embedding cost model, independent of output codec')


def profile_from_args(args):
    return select_profile(args.profile, args.stc, args.rate, args.delta, args.cost)
