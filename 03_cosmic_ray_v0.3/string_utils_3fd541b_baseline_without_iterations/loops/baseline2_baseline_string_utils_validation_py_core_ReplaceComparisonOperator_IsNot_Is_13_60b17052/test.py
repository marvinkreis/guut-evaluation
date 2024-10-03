from string_utils.validation import contains_html

def test_contains_html():
    # Test when input contains HTML tags
    input_string_with_html = "my string is <strong>bold</strong>"
    assert contains_html(input_string_with_html) == True, "Test failed: Expected True for string containing HTML tags."
    
    # Test when input does not contain HTML tags
    input_string_without_html = "my string is not bold"
    assert contains_html(input_string_without_html) == False, "Test failed: Expected False for string without HTML tags."