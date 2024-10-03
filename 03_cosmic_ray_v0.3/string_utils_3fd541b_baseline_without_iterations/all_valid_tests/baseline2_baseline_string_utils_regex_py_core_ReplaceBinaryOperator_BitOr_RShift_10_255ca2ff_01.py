from string_utils._regex import PRETTIFY_RE

def test__prettify_re():
    test_string = 'This is a test string with (some text) inside parentheses.'
    
    # The regex should match the parentheses and the text inside them
    matches = PRETTIFY_RE['SPACES_INSIDE'].findall(test_string)
    
    # With the correct code, there should be one match for the text inside the parentheses
    assert matches == ['some text'], f"Expected ['some text'], got {matches}"