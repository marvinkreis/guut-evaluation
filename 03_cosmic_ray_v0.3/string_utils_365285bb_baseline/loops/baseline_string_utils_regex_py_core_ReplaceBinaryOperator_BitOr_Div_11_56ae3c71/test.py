from string_utils._regex import PRETTIFY_RE

def test__prettify_re():
    # Input strings to test the PRETTIFY_RE regex patterns
    quoted_text = '"Hello, World!"'
    parenthesis_text = '(Hello, World!)'
    
    # Testing if the regex can match quoted text
    match_quoted = PRETTIFY_RE['SPACES_INSIDE'].search(quoted_text)
    assert match_quoted is not None, "Test failed for quoted text"
    
    # Testing if the regex can match text in parentheses
    match_parenthesis = PRETTIFY_RE['SPACES_INSIDE'].search(parenthesis_text)
    assert match_parenthesis is not None, "Test failed for parenthesis text"
    
    # Additional assertion to ensure that a text segment that has neither does not match
    non_matching_text = 'No special characters here.'
    match_non_matching = PRETTIFY_RE['SPACES_INSIDE'].search(non_matching_text)
    assert match_non_matching is None, "Test failed for non-matching text"