from string_utils.validation import is_integer

def test__is_integer():
    """Mutant changes 'and' to 'or' in is_integer, causing decimals to be incorrectly validated as integers."""
    test_input = "42.0"
    assert is_integer(test_input) is False, "is_integer must return False for decimal strings"