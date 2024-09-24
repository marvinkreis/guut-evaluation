from string_utils.manipulation import roman_encode

def test__roman_encode_zero():
    """Zero should raise ValueError in the correct implementation, but the mutant can process it incorrectly."""
    try:
        output = roman_encode(0)
        result = True  # If it doesn't raise, consider it a failed assertion below
    except ValueError:
        result = False  # Expected behavior gives this test a fall-through for correctness
    
    assert result is False, "Expected ValueError for roman_encode(0) but no error was raised."