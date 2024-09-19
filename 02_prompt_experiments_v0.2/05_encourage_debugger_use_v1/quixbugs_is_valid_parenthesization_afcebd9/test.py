from is_valid_parenthesization import is_valid_parenthesization

def test__is_valid_parenthesization():
    """The mutant changes the return value of is_valid_parenthesization to True regardless of input."""
    output = is_valid_parenthesization("((()))(")
    assert output is False, "is_valid_parenthesization must return False for unbalanced parentheses"