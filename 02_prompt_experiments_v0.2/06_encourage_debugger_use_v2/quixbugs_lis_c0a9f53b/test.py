from lis import lis

def test__lis():
    """The change in the longest assignment logic introduced an off-by-one error in the mutant."""
    output = lis([3, 2, 5, 6, 3, 7, 2, 8])
    assert output == 5, "Longest increasing subsequence length must be 5"