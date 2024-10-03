from string_utils.manipulation import prettify

def test_prettify_mutant_killing():
    """
    This test verifies that the prettify function raises a TypeError with the mutant
    due to an invalid placeholder key generation, while the baseline should return 
    the correctly formatted string.
    """
    input_string = 'This is a test. Please contact me at hello@example.com or visit my website at http://example.com.'
    
    # Expect the baseline to succeed
    output = prettify(input_string)
    assert isinstance(output, str), "Output should be a string"
    
    # The mutant should throw an error when running the code
    try:
        output = prettify(input_string)
    except Exception as e:
        assert isinstance(e, TypeError), f"Expected TypeError, got {type(e).__name__}"