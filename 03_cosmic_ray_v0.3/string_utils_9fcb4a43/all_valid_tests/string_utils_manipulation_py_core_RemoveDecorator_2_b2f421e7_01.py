from string_utils.manipulation import roman_decode

def test__roman_decode():
    """The mutant code has removed the __index_for_sign() method, causing a failure on valid Roman numeral input."""
    
    # Assert that the correct code returns the expected integer
    output = roman_decode('X')
    assert output == 10, "roman_decode should convert 'X' to 10"
    
    # Testing the mutant: Capture any exception that may arise.
    try:
        mutant_output = roman_decode('X')
        assert False, "Expected an exception from the mutant code when decoding 'X'. It should fail without producing a valid output."
    except Exception:
        # We expect that the mutant raises an exception, so the test passes here.
        pass