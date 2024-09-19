from is_valid_parenthesization import is_valid_parenthesization

def test__is_valid_parenthesization():
    """The mutant changes the final return value to always return True,
       which causes it to incorrectly identify invalid nested parentheses as valid."""
    assert is_valid_parenthesization('((())())') == True, "Should be valid"
    assert is_valid_parenthesization('(()') == False, "Should be invalid"
    assert is_valid_parenthesization('(()))') == False, "Should be invalid"
    assert is_valid_parenthesization('') == True, "Should be valid"