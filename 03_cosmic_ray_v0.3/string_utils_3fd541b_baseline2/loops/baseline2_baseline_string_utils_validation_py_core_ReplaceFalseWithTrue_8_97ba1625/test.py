from string_utils.validation import is_snake_case

def test__is_snake_case():
    # Testing valid snake case
    assert is_snake_case('valid_snake_case') == True, "Expected True for valid snake case."
    
    # Testing invalid snake case
    assert is_snake_case('InvalidSnakeCase') == False, "Expected False for invalid snake case (CamelCase)."
    assert is_snake_case('invalid-snake-case') == False, "Expected False for invalid snake case (invalid separator defined)."
    assert is_snake_case('snake_case_123') == True, "Expected True for valid snake case with numbers."
    
    # Testing invalid input
    assert is_snake_case('') == False, "Expected False for empty string."
    assert is_snake_case('123_snake_case') == False, "Expected False for string starting with number."