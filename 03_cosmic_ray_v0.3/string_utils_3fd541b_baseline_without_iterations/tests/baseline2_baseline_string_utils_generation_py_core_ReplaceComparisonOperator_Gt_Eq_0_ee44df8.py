from string_utils.generation import roman_range

def test_roman_range():
    # Test to ensure that requesting roman_range with 3999 works correctly
    result = list(roman_range(3999))
    assert result[-1] == 'MMMCMXCIX', "The last Roman numeral should be 'MMMCMXCIX' for stop=3999"
    
    # Test to check that using an invalid high value (beyond 3999) raises ValueError
    try:
        list(roman_range(4000))
        assert False, "Expected a ValueError when using stop=4000, but no exception was raised."
    except ValueError:
        pass  # Expected behavior
    
    # Test to check that using an invalid low value (0) raises ValueError
    try:
        list(roman_range(0))
        assert False, "Expected a ValueError when using stop=0, but no exception was raised."
    except ValueError:
        pass  # Expected behavior