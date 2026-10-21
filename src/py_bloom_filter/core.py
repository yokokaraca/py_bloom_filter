"""Core Bloom filter implementation.

Design choices
---------------
* ``sha256`` is used for hashing rather than the classic ``fnv``/``murmur`` pair,
  because it is available in the standard library on every supported Python version
  and behaves identically across platforms.  Its cost is acceptable given that this
  library targets correctness and portability over raw throughput.
* Double hashing (Kirsch and Mitzenmacher, 2005) is used to derive ``k`` distinct
  hash positions from a single digest, so we only pay one SHA-256 per lookup.
* The bit array is a ``bytearray``: one bit per cell, packed eight to a byte.  This
  keeps the structure compact without depending on any external libraries.
* Construction is explicit: the caller picks ``capacity`` and ``error_rate`` and we
  size the filter accordingly.  We do not implement automatic resizing, because an
  auto-resizing filter changes its false-positive guarantee mid-life and that is a
  semantic surprise we are not willing to ship.
"""

from __future__ import annotations

import hashlib
import math
from typing import Iterable, Iterator


_MAX_BITS = 8 * (2**31 - 1)


class BloomFilter:
    """A space-efficient set membership test with a tunable false positive rate.

    A Bloom filter can answer "is this item definitely not in the set?" or "might
    this item be in the set?".  It never reports a false negative, but may report a
    false positive with probability bounded by ``error_rate`` *provided* no more
    than ``capacity`` items have been inserted.

    Parameters
    ----------
    capacity:
        Maximum number of distinct items the filter is designed to hold while still
        honouring ``error_rate``.
    error_rate:
        Desired upper bound on the false positive probability once ``capacity``
        items have been inserted.  Must be in the open interval (0, 1).
    """

    __slots__ = ("capacity", "error_rate", "num_bits", "num_hashes", "_bits", "_count")

    def __init__(self, capacity: int, error_rate: float) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool):
            raise TypeError("capacity must be an int")
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if not isinstance(error_rate, (int, float)) or isinstance(error_rate, bool):
            raise TypeError("error_rate must be a real number")
        if not (0.0 < float(error_rate) < 1.0):
            raise ValueError("error_rate must satisfy 0 < error_rate < 1")

        m, k = _optimal_params(capacity, float(error_rate))
        self.capacity: int = capacity
        self.error_rate: float = float(error_rate)
        self.num_bits: int = m
        self.num_hashes: int = k
        self._bits: bytearray = bytearray((m + 7) // 8)
        self._count: int = 0

    def _positions(self, key: bytes) -> Iterator[int]:
        """Yield the ``num_hashes`` bit positions for ``key`` using double hashing.

        Double hashing derives ``k`` positions from a single digest as
        ``h1 + i * h2 mod m``.  We re-mix the seed for the second hash with a
        constant offset so that ``h1`` and ``h2`` are unlikely to collide for
        small or low-entropy inputs; the case ``h2 == 0`` is detected and the
        second hash is replaced by 1 to guarantee the probe sequence advances.
        """
        m = self.num_bits
        digest = hashlib.sha256(key).digest()
        h1 = int.from_bytes(digest[:8], "big")
        h2 = int.from_bytes(digest[8:16], "big")
        if h2 % m == 0:
            h2 = 1
        for i in range(self.num_hashes):
            yield (h1 + i * h2) % m

    def add(self, item: object) -> None:
        """Insert ``item`` into the filter.

        ``item`` is serialized with ``repr()`` so the filter can accept arbitrary
        Python objects.  This is intentional: restricting to ``bytes`` would force
        every caller to invent a serialization, and identical objects of the same
        type produce identical reprs in practice.  If you need a specific encoding,
        convert to ``bytes`` first and pass that.
        """
        key = repr(item).encode("utf-8")
        for pos in self._positions(key):
            self._bits[pos >> 3] |= 1 << (pos & 7)
        self._count += 1

    def __contains__(self, item: object) -> bool:
        """Return ``True`` if ``item`` may be in the filter, ``False`` if it is not."""
        key = repr(item).encode("utf-8")
        bits = self._bits
        for pos in self._positions(key):
            if not (bits[pos >> 3] & (1 << (pos & 7))):
                return False
        return True

    def __len__(self) -> int:
        """Return the number of items inserted.

        This is the count of ``add`` calls, not the number of set bits; it is an
        upper bound on the number of distinct items because duplicate inserts are
        not detected.
        """
        return self._count

    def __getstate__(self) -> dict:
        """Support ``copy`` and ``pickle``.

        Because ``__slots__`` is used, the default pickling path would skip our
        attributes unless we define this hook explicitly.
        """
        return {slot: getattr(self, slot) for slot in self.__slots__}

    def __setstate__(self, state: dict) -> None:
        for slot in self.__slots__:
            setattr(self, slot, state[slot])

    def approx_false_positive_rate(self) -> float:
        """Estimate the current false positive probability.

        Uses the standard formula ``(1 - exp(-kn/m))**k`` evaluated at the current
        insertion count.  This is an estimate, not a guarantee; once ``len(self)``
        exceeds ``capacity`` the actual rate will climb above ``error_rate``.
        """
        if self.num_bits == 0:
            return 0.0
        exponent = -self.num_hashes * self._count / self.num_bits
        # math.exp is used rather than the ``**`` operator on a float base so that
        # the large intermediate ``kn/m`` does not underflow to zero prematurely.
        return (1.0 - math.exp(exponent)) ** self.num_hashes

    @classmethod
    def from_iterable(
        cls, items: Iterable[object], capacity: int, error_rate: float
    ) -> "BloomFilter":
        """Build a filter pre-loaded with every element of ``items``."""
        bf = cls(capacity=capacity, error_rate=error_rate)
        for item in items:
            bf.add(item)
        return bf


def _optimal_params(capacity: int, error_rate: float) -> tuple[int, int]:
    """Return ``(num_bits, num_hashes)`` for the given capacity and error rate.

    The closed forms are ``m = -n ln(p) / (ln 2)^2`` and ``k = (m/n) ln 2``.
    We round ``m`` up to the nearest bit and ``k`` to the nearest positive integer,
    then validate against a sane hard limit so a pathological request does not try
    to allocate gigabytes of memory.
    """
    n = capacity
    p = error_rate
    m = math.ceil(-(n * math.log(p)) / (math.log(2) ** 2))
    if m <= 0:
        m = 1
    if m > _MAX_BITS:
        raise ValueError(
            f"required bit array size {m} exceeds the supported maximum "
            f"({_MAX_BITS}); reduce capacity or loosen error_rate"
        )
    k = max(1, round((m / n) * math.log(2)))
    return m, k
