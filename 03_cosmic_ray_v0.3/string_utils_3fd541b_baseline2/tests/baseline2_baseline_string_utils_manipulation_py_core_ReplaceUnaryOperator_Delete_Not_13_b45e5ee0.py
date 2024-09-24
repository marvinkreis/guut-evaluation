from string_utils.manipulation import strip_html
from string_utils.errors import InvalidInputError

def test_strip_html_invalid_input():
    # Test with an invalid input (not a string)
    invalid_input = 12345  # This is an integer and should raise an error
    try:
        strip_html(invalid_input)
    except InvalidInputError:
        pass  # We expect this exception, as the input is invalid
    else:
        assert False, "Expected InvalidInputError for non-string input"
        
    # Test with valid input just to ensure the function behaves correctly
    valid_input = "<div>Hello</div>"
    output = strip_html(valid_input)
    assert output == "", "Expected empty string after stripping HTML tags"