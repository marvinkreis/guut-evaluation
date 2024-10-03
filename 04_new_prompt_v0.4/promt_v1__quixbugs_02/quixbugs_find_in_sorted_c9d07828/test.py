from find_in_sorted import find_in_sorted

def test_find_in_sorted_mutant_killing():
    """
    Test the find_in_sorted function using a value greater than 
    the highest in the array. The input 1000 will cause the mutant 
    to enter an infinite recursion, while the baseline will return -1.
    """
    output = find_in_sorted([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 1000)
    assert output == -1, f"Expected -1, got {output}"