from max_sublist_sum import max_sublist_sum

def test__max_sublist_sum_kill_mutant():
    """
    This test checks for the maximum sublist sum when the input contains both positive
    and negative integers. The input [-1, 2, 3, -2, 5] has a maximum sublist of 
    [2, 3, -2, 5] which sums to 8, while the mutant will produce 7 due to the incorrect 
    accumulation of negative values into max_ending_here.
    """
    arr = [-1, 2, 3, -2, 5]
    output = max_sublist_sum(arr)
    assert output == 8