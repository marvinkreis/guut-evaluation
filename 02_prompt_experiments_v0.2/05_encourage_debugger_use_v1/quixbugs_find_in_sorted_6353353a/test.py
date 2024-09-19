from find_in_sorted import find_in_sorted

def test__find_in_sorted():
    """The mutant changes the behavior of binary search such that it can enter an infinite loop for missing elements."""
    arr = [3, 4, 5, 5, 5, 5, 6]
    x = 7  # 7 is not in the array
    output = find_in_sorted(arr, x)
    # Verify that the output is -1, indicating the element is not found
    assert output == -1, "find_in_sorted must return -1 when the element is not found."