from find_first_in_sorted import find_first_in_sorted

def test__find_first_in_sorted():
    """The change from 'while lo < hi' to 'while lo <= hi' causes the mutant to raise an IndexError for absent values."""
    # Testing with a value that is not present in the sorted list
    output = find_first_in_sorted([3, 4, 5, 5, 5, 5, 6], 7)
    assert output == -1, "The function should return -1 when the value is not present."