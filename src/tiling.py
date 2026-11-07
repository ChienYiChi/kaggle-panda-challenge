"""Small, dependency-free helpers for splitting work across processes."""


def split_ranges(n_items, n_parts):
    """Split ``range(n_items)`` into ``n_parts`` contiguous, inclusive ``(start, end)`` ranges.

    Every index appears in exactly one range and range sizes differ by at most one.
    When there are more parts than items, the trailing ranges are empty (``start > end``).
    """
    if n_parts < 1:
        raise ValueError("n_parts must be >= 1")
    base, extra = divmod(max(n_items, 0), n_parts)
    ranges, start = [], 0
    for i in range(n_parts):
        size = base + (1 if i < extra else 0)
        ranges.append((start, start + size - 1))
        start += size
    return ranges
