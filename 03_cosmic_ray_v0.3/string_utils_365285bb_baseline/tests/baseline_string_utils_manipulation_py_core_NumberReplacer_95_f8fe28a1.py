from string_utils.manipulation import snake_case_to_camel

def test_snake_case_to_camel():
    # Test with upper_case_first as True (should capitalize the first token)
    result = snake_case_to_camel('the_snake_is_green')
    assert result == 'TheSnakeIsGreen', f"Expected 'TheSnakeIsGreen' but got '{result}'"

    # Test with upper_case_first as False (should not capitalize the first token)
    result = snake_case_to_camel('this_is_a_test', upper_case_first=False)
    assert result == 'thisIsATest', f"Expected 'thisIsATest' but got '{result}'"

    # Test with upper_case_first as False and last token affecting outcome
    result = snake_case_to_camel('example_test_case', upper_case_first=False)
    assert result == 'exampleTestCase', f"Expected 'exampleTestCase' but got '{result}'"

    # Test with single word (should remain unchanged)
    result = snake_case_to_camel('singleword', upper_case_first=False)
    assert result == 'singleword', f"Expected 'singleword' but got '{result}'"

    # Test with multiple spaces in the input (should be handled)
    input_with_spaces = '   test_case   '
    result = snake_case_to_camel(input_with_spaces.strip())
    assert result == 'TestCase', f"Expected 'TestCase' but got '{result}'"