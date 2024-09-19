from possible_change import possible_change

def test__possible_change():
    """The mutant should raise an exception or produce incorrect output
    when the list of coins is empty while the original should return 0."""
    
    # Test with empty coins list and a positive total
    try:
        output = possible_change([], 5)
        assert output == 0, "Expected output is 0 for empty coins with a positive total"
    except Exception:
        assert False, "The original code raised an exception whereas it should return 0"
    
    # Test with a single denomination coin equal to the total
    try:
        output = possible_change([1], 1)
        assert output == 1, "Expected output is 1 for a single coin matching the total"
    except Exception:
        assert False, "The original code raised an exception for valid input"

# Call the test
test__possible_change()