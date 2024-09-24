from string_utils.manipulation import snake_case_to_camel

def test__snake_case_to_camel():
    # This should return 'TheSnakeIsGreen' when proper snake_case is provided
    result_correct = snake_case_to_camel('the_snake_is_green')
    assert result_correct == 'TheSnakeIsGreen', f"Expected 'TheSnakeIsGreen' but got {result_correct}"
    
    # This should return 'theSnakeIsGreen' when the first letter is to be in lowercase
    result_lower_first = snake_case_to_camel('the_snake_is_green', upper_case_first=False)
    assert result_lower_first == 'theSnakeIsGreen', f"Expected 'theSnakeIsGreen' but got {result_lower_first}"

    # If the mutant is present, this will lead to incorrect behavior
    # The first token ('the') should not be replaced by the last token ('green')

    # Check for an unexpected case that would expose the mutant:
    result_mutant = snake_case_to_camel('cat_dog')
    assert result_mutant == 'CatDog', f"Expected 'CatDog' but got {result_mutant}"

# Running the test
test__snake_case_to_camel()