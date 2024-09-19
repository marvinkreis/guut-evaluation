from kth import kth

def test__kth():
    """The mutant omits subtracting num_lessoreq when calculating the index for the recursive call, causing errors for certain values of k."""
    output = kth([3, 1, 4, 2], 2)
    assert output == 3, "The 2nd lowest element must be 3."
    
    output = kth([3, 1, 4, 2], 3)
    assert output == 4, "The 3rd lowest element must be 4."