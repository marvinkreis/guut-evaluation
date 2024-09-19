from lis import lis

def test__lis():
    """The mutant changes longest calculation, which leads to incorrect results."""
    output = lis([4, 1, 5, 3, 7, 6, 2])
    assert output == 3, "Should return length of the longest increasing subsequence"