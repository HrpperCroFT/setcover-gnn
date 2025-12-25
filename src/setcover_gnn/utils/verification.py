from typing import List, Set, Tuple


def greedy_set_cover(n: int, sets: List[Set[int]]) -> List[int]:
    """
    Greedy algorithm for Set Cover as baseline.
    
    Args:
        n: Number of elements
        sets: List of subsets
        
    Returns:
        Bitstring indicating selected subsets
    """
    assert set.union(*sets) == set(range(1, n + 1)), \
        "Some elements don't exist in subsets"
    
    n_sets = len(sets)
    bitstring = [0] * n_sets
    covered = set()
    
    while len(covered) < n:
        best_index = -1
        best_gain = -1
        
        for i in range(n_sets):
            if bitstring[i] == 0:
                gain = len(sets[i] - covered)
                if gain > best_gain:
                    best_gain = gain
                    best_index = i
        
        if best_index == -1:
            break
        
        bitstring[best_index] = 1
        covered.update(sets[best_index])
    
    return bitstring


def verify_set_cover(
    n: int, 
    sets: List[Set[int]], 
    bitstring: List[int]
) -> Tuple[bool, int]:
    """
    Verify if a solution is a valid set cover.
    
    Args:
        n: Number of elements
        sets: List of subsets
        bitstring: Solution vector
        
    Returns:
        Tuple of (is_valid_cover, number_of_selected_sets)
    """
    n_sets = len(sets)
    assert len(bitstring) == n_sets, \
        f"Lengths mismatch: sets={n_sets}, bitstring={len(bitstring)}"
    
    universal = set(range(1, n + 1))
    covered = set()
    selected_count = 0
    
    for i in range(n_sets):
        if bitstring[i] == 1:
            selected_count += 1
            covered.update(sets[i])
    
    is_cover = (covered == universal)
    return is_cover, selected_count