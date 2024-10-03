from max_sublist_sum import max_sublist_sum

def test__max_sublist_sum_kill_mutant():
    """
    Test to confirm that the max_sublist_sum function correctly calculates the maximum sum 
    of contiguous sublists, especially when negative values are present. The input [4, -5, 2, 1, -1, 3] 
    should yield a maximum sublist sum of 5 when run against the baseline and fail on the mutant.
    """
    arr = [4, -5, 2, 1, -1, 3]
    output = max_sublist_sum(arr)
    assert output == 5  # Expectation based on the logic of the baseline implementation