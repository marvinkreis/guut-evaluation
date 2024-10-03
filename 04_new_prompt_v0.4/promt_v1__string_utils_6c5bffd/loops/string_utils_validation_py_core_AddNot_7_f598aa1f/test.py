from string_utils.validation import is_url

def test__is_url():
    """
    Test whether the is_url function behaves correctly with allowed_schemes.
    The input is a valid URL (http://www.example.com). The baseline should return True for both an empty list
    and None for allowed_schemes, whereas the mutant will return False for the empty list (killing the mutant).
    """
    valid_url = 'http://www.example.com'
    
    # Test with an empty list of allowed schemes
    output_empty = is_url(valid_url, [])
    assert output_empty is True, f"Expected True, got {output_empty}"

    # Test with None allowed schemes, expecting an exception in the mutant
    try:
        output_none = is_url(valid_url, None)
        assert output_none is True, f"Expected True, got {output_none}"
    except TypeError:
        print("Caught TypeError as expected with None allowed schemes.")