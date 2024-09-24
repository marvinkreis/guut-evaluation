from string_utils.manipulation import prettify

def test__prettify():
    """Changing the replacement logic in __StringFormatter could lead to incorrect restoration of emails."""
    input_string = 'Here is a sample text with emails test@example.com and URLs https://example.com.'
    output = prettify(input_string)
    assert 'test@example.com' in output, "prettify must correctly restore emails"