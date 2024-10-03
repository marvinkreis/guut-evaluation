from string_utils._regex import PRETTIFY_RE

def test__prettify_re():
    # Test to detect the mutant by inputting text that includes quoted text and text in brackets
    input_text = 'This is a test "quoted text" and (text in brackets).'
    
    # Find matches based on the original regex
    matches = PRETTIFY_RE['SPACES_INSIDE'].findall(input_text)
    
    # The expected matches
    expected_matches = ['quoted text', 'text in brackets']
    
    # Assert that the matches found by the regex in the correct implementation matches the expectation
    assert matches == expected_matches, f'Expected {expected_matches} but got {matches}'
    
    # If the mutant is present, the regex will malfunction, and we can use the assertion to verify that
    # The purpose is to use this negative case to see the mutant fail.