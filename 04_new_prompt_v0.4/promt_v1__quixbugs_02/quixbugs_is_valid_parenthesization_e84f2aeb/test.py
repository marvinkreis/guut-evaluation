from is_valid_parenthesization import is_valid_parenthesization

def test__is_valid_parenthesization_kill_mutant():
    """
    Test whether the function correctly identifies improperly nested parentheses. The input '((())' should return False for the baseline but True for the mutant.
    """
    output = is_valid_parenthesization('((())')
    assert output == False