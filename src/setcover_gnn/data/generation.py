import random
from typing import Optional


def generate_set_cover_instance(
    n: int, num_sets: Optional[int] = None, coverage_factor: float = 1.5
) -> list[set[int]]:
    """Generates a random Set Cover instance.

    Args:
        n: Number of elements in universal set
        num_sets: Number of subsets (default: random between n/2 and 2n)
        coverage_factor: Controls density of coverage

    Returns:
        List of subsets
    """
    if num_sets is None:
        num_sets = random.randint((n + 1) // 2, 2 * n)

    subsets: list[set[int]] = [set() for _ in range(num_sets)]
    universal_set = set(range(1, n + 1))

    for i in range(num_sets):
        for j in range(n):
            if random.random() <= coverage_factor / num_sets:
                subsets[i].add(j + 1)

    not_exists = universal_set - set().union(*subsets)

    for num in not_exists:
        idx = random.randint(0, num_sets - 1)
        subsets[idx].add(num)

    random.shuffle(subsets)

    return subsets
