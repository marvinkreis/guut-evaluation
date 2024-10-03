from kheapsort import kheapsort

def test__kheapsort_kills_mutant():
    """
    Test kheapsort on a specific input where k is less than the length of the array.
    The input array is [3, 2, 1, 5, 4] with k=2. The expected result is [1, 2, 3, 4, 5].
    The mutant will fail this test because it incorrectly iterates over the entire array.
    """
    output = list(kheapsort([3, 2, 1, 5, 4], 2))
    assert output == [1, 2, 3, 4, 5], f"Expected [1, 2, 3, 4, 5], but got {output}"