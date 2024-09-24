from string_utils.manipulation import snake_case_to_camel

def test_snake_case_to_camel():
    # Straightforward cases that emphasize the importance of the first token.
    
    assert snake_case_to_camel('first_second') == 'FirstSecond'  # Expect: FirstSecond
    assert snake_case_to_camel('hello_world') == 'HelloWorld'  # Expect: HelloWorld
    assert snake_case_to_camel('example_test') == 'ExampleTest'  # Expect: ExampleTest
    assert snake_case_to_camel('single_case') == 'SingleCase'  # Expect: SingleCase
    assert snake_case_to_camel('another_case') == 'AnotherCase'  # Expect: AnotherCase
    
    # More complex cases with underscores
    assert snake_case_to_camel('one_two') == 'OneTwo'  # Expect: OneTwo
    assert snake_case_to_camel('multiple_parts_example') == 'MultiplePartsExample'  # Expect: MultiplePartsExample
    assert snake_case_to_camel('several_tokens_in_a_row') == 'SeveralTokensInARow'  # Expect: SeveralTokensInARow

    # Edge cases with underscores
    assert snake_case_to_camel('leading_underscore') == 'LeadingUnderscore'  # Expect: LeadingUnderscore
    assert snake_case_to_camel('trailing_underscore_') == 'TrailingUnderscore'  # Expect: TrailingUnderscore
    assert snake_case_to_camel('double__underscore_case') == 'DoubleUnderscoreCase'  # Expect: DoubleUnderscoreCase

    # Assert on empty input
    assert snake_case_to_camel('') == ''  # Expect empty input to return empty

# Run the test
test_snake_case_to_camel()