from lis import lis

def test__lis():
    """The mutant modifies the logic such that it cannot accurately calculate the longest increasing subsequence."""
    assert lis([4, 1, 5, 3, 7, 6, 2]) == 3, "Expected length of LIS is 3"
    assert lis([10, 22, 9, 33, 21, 50, 41, 60, 80]) == 6, "Expected length of LIS is 6"

test__lis()