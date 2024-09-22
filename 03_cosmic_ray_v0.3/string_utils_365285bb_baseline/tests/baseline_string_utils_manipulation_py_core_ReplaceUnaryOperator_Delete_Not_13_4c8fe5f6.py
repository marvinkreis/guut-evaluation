from string_utils.manipulation import strip_html
from string_utils.errors import InvalidInputError

def test_strip_html():
    # Test with valid string input
    result = strip_html('test: <a href="foo/bar">click here</a>')
    assert result == 'test: ', f"Expected 'test: ', got '{result}'"

    # Test with invalid input: integer should raise InvalidInputError
    try:
        strip_html(12345)
        assert False, "Expected InvalidInputError for non-string input"
    except InvalidInputError:
        pass  # This is expected

    # Test with invalid input: None should raise InvalidInputError
    try:
        strip_html(None)
        assert False, "Expected InvalidInputError for None input"
    except InvalidInputError:
        pass  # This is expected