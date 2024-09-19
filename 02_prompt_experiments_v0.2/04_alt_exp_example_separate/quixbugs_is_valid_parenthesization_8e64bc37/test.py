from is_valid_parenthesization import is_valid_parenthesization

def test__is_valid_parenthesization():
    """The mutant incorrectly returns True for invalid parenthesis strings."""
    output = is_valid_parenthesization('(())(')
    assert output is False, "Expected False for invalid nesting, but got True"