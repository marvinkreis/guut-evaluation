from string_utils.manipulation import prettify

def test__prettify():
    """Even though the surrounding space handling seems consistent, the mutated version may yield unexpected behavior in overlapping scenarios."""
    input_string = '  hello  world  '
    assert prettify(input_string) == 'Hello world', f"Expected 'Hello world', got {prettify(input_string)}"
    
    input_string = 'hello world   '
    assert prettify(input_string) == 'Hello world', f"Expected 'Hello world', got {prettify(input_string)}"
    
    input_string = '  hello   '
    assert prettify(input_string) == 'Hello', f"Expected 'Hello', got {prettify(input_string)}"
    
    input_string = '  hello. world  '
    assert prettify(input_string) == 'Hello. World', f"Expected 'Hello. World', got {prettify(input_string)}"
    
    input_string = 'hello   . world'
    assert prettify(input_string) == 'Hello. World', f"Expected 'Hello. World', got {prettify(input_string)}"