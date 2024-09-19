from find_first_in_sorted import find_first_in_sorted

def test__find_first_in_sorted():
    """Changing 'while lo < hi' to 'while lo <= hi' in find_first_in_sorted may cause an index error."""
    arr = [3, 4, 5, 5, 5, 5, 6]
    x = 7  # Element does not exist
    output = find_first_in_sorted(arr, x)
    
    # The correct output should be -1 as the element does not exist
    assert output == -1, "Expected output for a missing element should be -1"