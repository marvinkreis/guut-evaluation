from find_first_in_sorted import find_first_in_sorted

def test__find_first_in_sorted():
    """The mutant causes an IndexError when the target is greater than all elements in the array."""
    arr = [1, 2, 3, 4, 5]
    x = 6  # target larger than any element in arr
    output = find_first_in_sorted(arr, x)
    assert output == -1, "Expected -1 since 6 is not in the array"