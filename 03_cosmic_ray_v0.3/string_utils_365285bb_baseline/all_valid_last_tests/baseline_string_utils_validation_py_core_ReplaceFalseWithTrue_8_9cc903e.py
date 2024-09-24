from string_utils.validation import is_snake_case

def test_is_snake_case():
    # Test case for an invalid snake case string
    invalid_snake_case_string = 'InvalidString'  # No underscores, should return False
    result = is_snake_case(invalid_snake_case_string)
    assert result == False, f"Expected False but got {result}"

    # Test case for a valid snake case string
    valid_snake_case_string = 'valid_snake_case'  # Valid snake case, should return True
    result = is_snake_case(valid_snake_case_string)
    assert result == True, f"Expected True but got {result}"

    # Additional test case for another invalid input which should not be snake case
    another_invalid_string = '123_invalid_snake'  # Starts with numbers, should return False
    result = is_snake_case(another_invalid_string)
    assert result == False, f"Expected False but got {result}"

    # Test case for a string with invalid characters for snake_case
    invalid_characters_string = 'invalid@snake_case'  # Contains invalid '@', should return False
    result = is_snake_case(invalid_characters_string)
    assert result == False, f"Expected False but got {result}"

# Call the test function
test_is_snake_case()