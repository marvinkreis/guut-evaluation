from string_utils.manipulation import snake_case_to_camel

def test_snake_case_to_camel():
    # Valid conversions from snake_case to CamelCase
    assert snake_case_to_camel('hello_world') == 'HelloWorld', "Expected 'HelloWorld'"
    assert snake_case_to_camel('sample_case_example') == 'SampleCaseExample', "Expected 'SampleCaseExample'"

    # Test mixed inputs that are valid but not snake_case
    assert snake_case_to_camel('MixedCaseString') == 'MixedCaseString', "Expected 'MixedCaseString'"
    
    # These should return themselves since they're not snake case
    assert snake_case_to_camel('123not_a_valid_case') == '123not_a_valid_case', "Expected '123not_a_valid_case'"
    assert snake_case_to_camel('not-snake-case') == 'not-snake-case', "Expected 'not-snake-case'"

    # Valid snake_case inputs mixed with numbers
    assert snake_case_to_camel('valid_123_case') == 'Valid123Case', "Expected 'Valid123Case'"
    assert snake_case_to_camel('input_with_numbers_456') == 'InputWithNumbers456', "Expected 'InputWithNumbers456'"

    # Edge cases: leading/trailing underscores
    assert snake_case_to_camel('_leading_underscore') == 'LeadingUnderscore', "Expected 'LeadingUnderscore'"
    assert snake_case_to_camel('trailing_underscore_') == 'TrailingUnderscore', "Expected 'TrailingUnderscore'"

    # Edge case: multiple underscores sequentially
    assert snake_case_to_camel('this____is_a_test') == 'ThisIsATest', "Expected 'ThisIsATest'"
    
    # Inputs that aren't valid for transformation
    assert snake_case_to_camel('invalid_case!') == 'invalid_case!', "Expected 'invalid_case!'"
    assert snake_case_to_camel('another_with_special_#_chars') == 'another_with_special_#_chars', "Expected 'another_with_special_#_chars'"

    # Edge Test: An input that should expose a mistake in mutant processing
    assert snake_case_to_camel('multiple___underscores') == 'MultipleUnderscores', "Expected 'MultipleUnderscores'"

# Run the test function
test_snake_case_to_camel()