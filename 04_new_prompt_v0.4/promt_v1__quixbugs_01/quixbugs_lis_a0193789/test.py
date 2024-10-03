from lis import lis

def test__lis_kill_mutant():
    """
    Test to check the length of the longest increasing subsequence.
    The input [1, 3, 6, 7, 9, 4, 10, 5, 6] has a longest increasing subsequence
    of length 6 ([1, 3, 6, 7, 9, 10]), which the mutant fails to compute, returning 5 instead.
    """
    arr = [1, 3, 6, 7, 9, 4, 10, 5, 6]  # Known output is 6
    output = lis(arr)
    assert output == 6  # Expecting the output to be 6