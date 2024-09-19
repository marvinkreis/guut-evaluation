from find_in_sorted import find_in_sorted

def test__find_in_sorted():
    """The mutant will cause a recursion error when searching for a non-existent element."""
    arr = [1, 2, 3, 4]
    x = 5
    output = find_in_sorted(arr, x)
    assert output == -1, "Expected -1 because the element is not in the array"