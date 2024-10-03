from kth import kth

def test__kth_kills_mutant():
    """
    This test verifies that the kth function returns the correct element in the array.
    Specifically, it checks for the case where k is rightmost in the range,
    leading the mutant to incorrectly execute a recursive call that results in an error.
    The test with k=5 on the array will expose this flaw, where the expected output is 20.
    """
    arr = [7, 10, 4, 3, 20, 15]
    k = 5  # The sorted array is [3, 4, 7, 10, 15, 20], so the 5th index is 20
    output = kth(arr, k)
    assert output == 20