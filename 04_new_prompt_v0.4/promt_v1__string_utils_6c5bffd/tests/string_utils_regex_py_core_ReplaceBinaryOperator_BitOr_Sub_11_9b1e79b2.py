from string_utils._regex import PRETTIFY_RE

def test__spaces_inside_regex():
    """
    Test that the 'SPACES_INSIDE' regex correctly captures quoted strings 
    and text in parentheses even when they contain new lines. The mutant 
    fails to run due to an incorrect modification that causes a ValueError 
    when the regex is compiled, demonstrating that the mutant is 
    improperly defined.
    """
    test_string = 'This is a test string with "quotes and \n new lines" and (parenthesis with \n new lines).'
    
    # This will only be successful if the regex compiles and works correctly
    spaces_inside_pattern = PRETTIFY_RE['SPACES_INSIDE']
    output = spaces_inside_pattern.findall(test_string)
    assert output == ['quotes and \n new lines', 'parenthesis with \n new lines']