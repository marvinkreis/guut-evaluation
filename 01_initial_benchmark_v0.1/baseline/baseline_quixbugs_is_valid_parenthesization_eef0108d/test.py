from is_valid_parenthesization import is_valid_parenthesization

def test_is_valid_parenthesization():
    # Test with a valid nested parentheses
    result = is_valid_parenthesization('((()()))()')
    assert result == True, f"Expected True, got {result}"
    
    # Test with a string of invalid parentheses
    result = is_valid_parenthesization(')()(')
    assert result == False, f"Expected False, got {result}"
    
    # Test with an empty string (which is also considered valid)
    result = is_valid_parenthesization('')
    assert result == True, f"Expected True, got {result}"
    
    # Test with a valid complex nested parentheses
    result = is_valid_parenthesization('((())())(())')
    assert result == True, f"Expected True, got {result}"
    
    # Test with a single open parenthesis
    result = is_valid_parenthesization('(')
    assert result == False, f"Expected False, got {result}"
    
    # Test with a single close parenthesis
    result = is_valid_parenthesization(')')
    assert result == False, f"Expected False, got {result}"

    # Test with a long valid sequence of parentheses
    result = is_valid_parenthesization('()()()()()()()()')
    assert result == True, f"Expected True, got {result}"