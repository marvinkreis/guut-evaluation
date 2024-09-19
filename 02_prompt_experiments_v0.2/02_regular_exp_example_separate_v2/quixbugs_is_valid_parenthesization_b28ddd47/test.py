from is_valid_parenthesization import is_valid_parenthesization

def test__is_valid_parenthesization():
    """The mutant's unconditional return True will cause it to incorrectly validate invalid parentheses."""
    assert is_valid_parenthesization('((()()))()') == True, "Should be True for valid input"
    assert is_valid_parenthesization(')()(') == False, "Should be False for invalid input"
    assert is_valid_parenthesization('(()') == False, "Should be False for invalid input"