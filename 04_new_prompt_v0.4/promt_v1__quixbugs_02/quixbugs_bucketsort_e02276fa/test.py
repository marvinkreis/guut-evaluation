from bucketsort import bucketsort

def test__bucketsort_kills_mutant():
    """
    Test that the bucketsort function sorts the array correctly.
    The input array [3, 1, 2, 0, 4] should be sorted to [0, 1, 2, 3, 4].
    The mutant will fail this test as it does not sort correctly.
    """
    arr = [3, 1, 2, 0, 4]
    k = 5
    output = bucketsort(arr, k)
    assert output == [0, 1, 2, 3, 4], f"Expected [0, 1, 2, 3, 4], but got {output}"