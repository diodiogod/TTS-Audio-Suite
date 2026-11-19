"""Per-table bucket counts of the condition rows (port of minisgl's CONDITION_BUCKET_COUNTS).

The cond3 finetunes share one bucket count for their tables (params ``condition_n_buckets``; the id equal to the count
is the unknown row). The rate30 finetunes widen one table (params ``condition_bucket_counts`` [10, 10, 30]): table i
has counts[i] + 1 rows and its unknown id is counts[i]. Without the field every table has ``condition_n_buckets``.
"""

from __future__ import annotations

from typing import Sequence


def bucket_counts(count: int, n_buckets: int, counts: Sequence[int] | None = None) -> tuple[int, ...]:
    """One bucket count per condition table: ``counts`` when the checkpoint carries them, else ``n_buckets`` for every
    table. A table's unknown id is its count."""
    if count <= 0:
        return ()
    if not counts:
        return (int(n_buckets),) * count
    if len(counts) != count:
        raise ValueError(f"condition_bucket_counts has {len(counts)} entries for {count} condition rows")
    if any(int(c) < 2 for c in counts):
        raise ValueError(f"condition_bucket_counts must be integers >= 2: {list(counts)}")
    return tuple(int(c) for c in counts)


def check_ids(ids, counts: Sequence[int], what: str = "condition_buckets") -> list[int]:
    """Validate one id per table (0..count, count = unknown) and return them as ints."""
    try:
        ids = [int(b) for b in ids]
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be integers") from None
    counts = [int(u) for u in counts]
    if len(ids) != len(counts) or any(not 0 <= b <= u for b, u in zip(ids, counts)):
        raise ValueError(
            f"{what} needs {len(counts)} ids, each in [0, count] for the bucket counts {counts} (count = unknown)"
        )
    return ids
