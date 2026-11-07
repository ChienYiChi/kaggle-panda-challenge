import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tiling import split_ranges  # noqa: E402


def covered(ranges):
    return [i for start, end in ranges for i in range(start, end + 1)]


class SplitRangesTest(unittest.TestCase):
    def test_every_index_exactly_once(self):
        for n_items in (0, 1, 2, 7, 10, 100, 10616):
            for n_parts in (1, 2, 3, 4, 8, 16):
                with self.subTest(n_items=n_items, n_parts=n_parts):
                    self.assertEqual(covered(split_ranges(n_items, n_parts)), list(range(n_items)))

    def test_sizes_differ_by_at_most_one(self):
        sizes = [end - start + 1 for start, end in split_ranges(10, 4)]
        self.assertEqual(sizes, [3, 3, 2, 2])

    def test_more_parts_than_items_gives_empty_ranges(self):
        ranges = split_ranges(2, 4)
        self.assertEqual([r for r in ranges if r[0] <= r[1]], [(0, 0), (1, 1)])
        self.assertTrue(all(start > end for start, end in ranges[2:]))

    def test_first_slide_is_not_skipped(self):
        # Regression: the previous partition started the first worker at index 1, so slide 0 was never tiled.
        self.assertEqual(split_ranges(10, 4)[0][0], 0)

    def test_rejects_zero_parts(self):
        with self.assertRaises(ValueError):
            split_ranges(10, 0)


if __name__ == "__main__":
    unittest.main()
