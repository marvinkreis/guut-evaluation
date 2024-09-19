from lis import lis

def test__lis():
    """The mutant does not consider the previously calculated longest length, 
    which leads to incorrect results for the longest increasing subsequence."""
    output = lis([4, 1, 5, 3, 7, 6, 2])
    assert output > 2, "The LIS function must find a longer increasing subsequence."