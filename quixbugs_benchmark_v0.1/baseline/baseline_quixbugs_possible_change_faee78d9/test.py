from possible_change import possible_change

def test_possible_change():
    # Test where there are no coins but a positive total is requested.
    # This should return 0 since we cannot make change.
    result = possible_change([], 5)
    assert result == 0, f"Expected 0 but got {result}"

    # Additional test case to check with no coins and zero total
    result = possible_change([], 0)
    assert result == 1, f"Expected 1 but got {result}"

    # Test with some coins and totals that should result in valid counts
    result = possible_change([1, 2, 5], 5)
    assert result == 4, f"Expected 4 but got {result}"

    # Test with coins but requesting a negative total
    result = possible_change([1, 2, 5], -1)
    assert result == 0, f"Expected 0 but got {result}"