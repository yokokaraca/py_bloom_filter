"""Tests for py_bloom_filter.core.BloomFilter.

These tests are deliberately explicit about the behaviour they check and avoid
asserting on timing, randomness, or floating-point equality.  Where a float is
involved, we compare against a bound or use ``math.isclose``.
"""

import math
import pickle
import unittest

from py_bloom_filter import BloomFilter


class TestConstruction(unittest.TestCase):
    def test_rejects_zero_capacity(self):
        with self.assertRaises(ValueError):
            BloomFilter(capacity=0, error_rate=0.01)

    def test_rejects_negative_capacity(self):
        with self.assertRaises(ValueError):
            BloomFilter(capacity=-10, error_rate=0.01)

    def test_rejects_non_int_capacity(self):
        with self.assertRaises(TypeError):
            BloomFilter(capacity=10.0, error_rate=0.01)

    def test_rejects_bool_capacity(self):
        with self.assertRaises(TypeError):
            BloomFilter(capacity=True, error_rate=0.01)

    def test_rejects_error_rate_zero(self):
        with self.assertRaises(ValueError):
            BloomFilter(capacity=100, error_rate=0.0)

    def test_rejects_error_rate_one(self):
        with self.assertRaises(ValueError):
            BloomFilter(capacity=100, error_rate=1.0)

    def test_rejects_error_rate_out_of_range(self):
        with self.assertRaises(ValueError):
            BloomFilter(capacity=100, error_rate=1.5)

    def test_rejects_bool_error_rate(self):
        with self.assertRaises(TypeError):
            BloomFilter(capacity=100, error_rate=True)

    def test_valid_construction_sets_fields(self):
        bf = BloomFilter(capacity=1000, error_rate=0.01)
        self.assertEqual(bf.capacity, 1000)
        self.assertAlmostEqual(bf.error_rate, 0.01)
        self.assertGreater(bf.num_bits, 0)
        self.assertGreaterEqual(bf.num_hashes, 1)
        self.assertEqual(len(bf), 0)


class TestMembership(unittest.TestCase):
    def test_absent_item_returns_false(self):
        bf = BloomFilter(capacity=100, error_rate=0.01)
        self.assertNotIn("missing", bf)

    def test_added_item_returns_true(self):
        bf = BloomFilter(capacity=100, error_rate=0.01)
        bf.add("present")
        self.assertIn("present", bf)

    def test_distinct_types_are_distinct_keys(self):
        bf = BloomFilter(capacity=100, error_rate=0.01)
        bf.add(1)
        self.assertIn(1, bf)
        self.assertNotIn("1", bf)

    def test_bytes_and_str_are_distinct(self):
        bf = BloomFilter(capacity=100, error_rate=0.01)
        bf.add(b"hello")
        self.assertIn(b"hello", bf)
        self.assertNotIn("hello", bf)

    def test_none_can_be_stored(self):
        bf = BloomFilter(capacity=100, error_rate=0.01)
        bf.add(None)
        self.assertIn(None, bf)

    def test_tuple_can_be_stored(self):
        bf = BloomFilter(capacity=100, error_rate=0.01)
        bf.add((1, 2, 3))
        self.assertIn((1, 2, 3), bf)

    def test_length_counts_inserts_not_distinct(self):
        bf = BloomFilter(capacity=100, error_rate=0.01)
        bf.add("x")
        bf.add("x")
        self.assertEqual(len(bf), 2)

    def test_from_iterable_preloads(self):
        bf = BloomFilter.from_iterable(["a", "b", "c"], capacity=100, error_rate=0.01)
        self.assertEqual(len(bf), 3)
        for item in ("a", "b", "c"):
            self.assertIn(item, bf)


class TestFalsePositiveBound(unittest.TestCase):
    def test_no_false_negatives_under_capacity(self):
        # Deterministic: we only assert the no-false-negative guarantee.
        bf = BloomFilter(capacity=200, error_rate=0.01)
        for i in range(200):
            bf.add(i)
        for i in range(200):
            self.assertIn(i, bf)

    def test_approx_rate_is_zero_when_empty(self):
        bf = BloomFilter(capacity=100, error_rate=0.01)
        self.assertEqual(bf.approx_false_positive_rate(), 0.0)

    def test_approx_rate_stays_near_target_at_capacity(self):
        bf = BloomFilter(capacity=1000, error_rate=0.01)
        for i in range(1000):
            bf.add(i)
        rate = bf.approx_false_positive_rate()
        # The classic formula can land slightly above the target due to rounding
        # of k; we allow a generous but deterministic margin.
        self.assertLess(rate, 0.05)


class TestPersistence(unittest.TestCase):
    def test_pickle_roundtrip_preserves_membership(self):
        bf = BloomFilter(capacity=100, error_rate=0.01)
        bf.add("keep")
        bf.add("this")
        data = pickle.dumps(bf)
        restored = pickle.loads(data)
        self.assertIn("keep", restored)
        self.assertIn("this", restored)
        self.assertEqual(len(restored), 2)
        self.assertEqual(restored.num_bits, bf.num_bits)
        self.assertEqual(restored.num_hashes, bf.num_hashes)


class TestSizing(unittest.TestCase):
    def test_smaller_error_rate_requires_more_bits(self):
        loose = BloomFilter(capacity=1000, error_rate=0.1)
        tight = BloomFilter(capacity=1000, error_rate=0.001)
        self.assertGreater(tight.num_bits, loose.num_bits)

    def test_larger_capacity_requires_more_bits(self):
        small = BloomFilter(capacity=100, error_rate=0.01)
        large = BloomFilter(capacity=10000, error_rate=0.01)
        self.assertGreater(large.num_bits, small.num_bits)

    def test_extreme_request_is_rejected(self):
        with self.assertRaises(ValueError):
            BloomFilter(capacity=10**18, error_rate=0.0001)


if __name__ == "__main__":
    unittest.main()
