from kheapsort import kheapsort

def test__kheapsort_kill_mutant():
    """
    Test if the k-heapsort function generates any duplicates in its output. 
    The input array [3, 2, 1, 5, 4] with k=2 should yield a unique sorted output 
    without duplicates from the baseline. The mutant, however, produces duplicates,
    allowing us to distinguish between the two versions.
    """
    arr = [3, 2, 1, 5, 4]
    k = 2
    output = list(kheapsort(arr, k))
    has_duplicates = len(output) != len(set(output))
    assert not has_duplicates