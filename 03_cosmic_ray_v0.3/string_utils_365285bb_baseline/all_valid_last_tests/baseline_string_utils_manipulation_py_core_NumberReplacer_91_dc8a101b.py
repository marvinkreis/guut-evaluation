from string_utils.manipulation import camel_case_to_snake

def test_basic_camel_case_conversion():
    # A straightforward camel case with two words
    input_string = "CamelCase"
    expected_output = "camel_case"
    
    result = camel_case_to_snake(input_string)
    assert result == expected_output, f"Expected '{expected_output}', but got '{result}'"

def test_multiple_camel_case_words():
    # More complex camel case
    input_string = "ThisIsCamelCase"
    expected_output = "this_is_camel_case"
    
    result = camel_case_to_snake(input_string)
    assert result == expected_output, f"Expected '{expected_output}', but got '{result}'"

def test_with_acronyms():
    # Test how it handles acronyms in camel case
    input_string = "APIResponseHandler"
    expected_output = "api_response_handler"
    
    result = camel_case_to_snake(input_string)
    assert result == expected_output, f"Expected '{expected_output}', but got '{result}'"

def test_numbers_with_camel_case():
    # Include numbers along with camel case
    input_string = "Camel2Case"
    expected_output = "camel2_case"
    
    result = camel_case_to_snake(input_string)
    assert result == expected_output, f"Expected '{expected_output}', but got '{result}'"

def test_edge_case_with_double_uppercase():
    # Check consecutive uppercase letters
    input_string = "HTTPRequest"
    expected_output = "http_request"
    
    result = camel_case_to_snake(input_string)
    assert result == expected_output, f"Expected '{expected_output}', but got '{result}'"