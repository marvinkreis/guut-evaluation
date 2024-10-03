from find_first_in_sorted import find_first_in_sorted

def test__find_first_in_sorted():
    """
    Test the function with a value that is not in the array. This should return -1 for the baseline but may cause an IndexError in the mutant due to the inclusive loop condition.
    """
    output = find_first_in_sorted([1, 2, 3, 4, 5], 6)
    assert output == -1  # Expecting -1 from the baseline