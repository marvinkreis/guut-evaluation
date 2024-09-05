from is_valid_parenthesization import is_valid_parenthesization

def test__is_valid_parenthesization():
    assert is_valid_parenthesization('(') == False, "Must return False for unmatched opening parenthesis"