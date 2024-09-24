from string_utils.validation import is_pangram

def test__is_pangram():
    """Changing the condition in is_pangram to check for is_full_string incorrectly rejects valid pangrams."""
    input_string = "The quick brown fox jumps over the lazy dog"
    assert is_pangram(input_string) == True, "is_pangram must return True for a valid pangram"