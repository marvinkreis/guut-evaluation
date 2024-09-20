from string_utils.validation import is_full_string

def mock_mutant_is_full_string(input_string):
    if input_string is None:
        raise AttributeError("NoneType object has no attribute 'strip'")
    return isinstance(input_string, str) and input_string.strip() != ''

def test__is_full_string():
    """Test different cases for is_full_string and check for mutant's behavior."""
    
    # Checking behavior for None
    assert not is_full_string(None), "Expected False for None input from correct implementation."
    print("Correct implementation handled None and returned False.")

    # Additional inputs
    assert not is_full_string(''), "Expected False for empty string input"
    assert not is_full_string(' '), "Expected False for whitespace only input"
    assert is_full_string('Hello'), "Expected True for non-empty input"
    assert is_full_string('   Hello   '), "Expected True for non-empty input with spaces"
    assert not is_full_string('  '), "Expected False for whitespace input"

    # Test the mutant behavior using the mock function
    print("Testing mock mutant implementation with None:")
    try:
        mock_mutant_is_full_string(None)  # Should raise an exception
        assert False, "Expected mock mutant to raise AttributeError for None"
    except AttributeError:
        print("Caught expected error in mock mutant implementation when passing None.")

# Call the test function to verify behavior
test__is_full_string()