from is_valid_parenthesization import is_valid_parenthesization

def test__is_valid_parenthesization():
    """The mutant fixes depth but should have reflected the actual balance of parentheses."""
    assert is_valid_parenthesization('(()') == False, "Mutant should return True, but it doesn't."