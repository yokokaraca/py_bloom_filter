# py_bloom_filter

A space-efficient set membership test with a tunable false positive rate, implemented in the Python standard library with zero third-party dependencies.

```python
from py_bloom_filter import BloomFilter

bf = BloomFilter(capacity=10_000, error_rate=0.01)
bf.add("user:42")

print("user:42" in bf)   # True
print("user:99" in bf)    # False (definitely not present, at least for now)
```

`BloomFilter(capacity, error_rate)` sizes the bit array and the number of hash functions for the requested false positive rate, assuming at most `capacity` distinct insertions. `add(item)` inserts any Python object (serialized via `repr`), `item in bf` tests membership, and `len(bf)` reports the number of `add` calls. The filter supports `pickle` and exposes `approx_false_positive_rate()` for introspection.

This exists for the common case where you want to cheaply reject work that is definitely not needed — for example, asking "might this key already be in the cache?" before paying for a database lookup. The trade-off is that membership is probabilistic: a `True` result means "possibly present" and the false positive rate climbs past the configured `error_rate` once you exceed `capacity`. There is no automatic resizing, because resizing silently changes the guarantee mid-life and that is a surprise we are not willing to ship.

The awkward edge: `len(bf)` counts insertions, not distinct items. Inserting the same key twice increments the counter both times, which is how bloom filters behave by construction but can surprise callers who expect set semantics.
