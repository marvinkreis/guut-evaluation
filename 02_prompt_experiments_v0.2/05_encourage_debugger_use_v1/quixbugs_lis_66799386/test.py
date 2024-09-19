from lis import lis

def test__lis():
    """The mutant improperly calculates the longest increasing subsequence length by specifically setting longest based on the new value rather than checking it against the maximum."""
    # First test case
    output1 = lis([4, 1, 5, 3, 7, 6, 2])
    assert output1 == 3, "The longest increasing subsequence should yield a length of 3"
    
    # Second test case
    output2 = lis([10, 22, 9, 33, 21, 50, 41, 60, 80])
    assert output2 == 6, "The longest increasing subsequence should yield a length of 6"