from string_utils._regex import PRETTIFY_RE

def test__PRETTIFY_RE():
    """The mutant changes 're.MULTILINE | re.DOTALL' to 're.MULTILINE * re.DOTALL', leading to incorrect behavior."""
    # Test input which includes spaces and structured format
    test_input = 'This is an example ;  that should match.'
    
    # Using the RIGHT_SPACE regex rule from the PRETTIFY_RE
    matches = PRETTIFY_RE['RIGHT_SPACE'].findall(test_input)
    
    assert len(matches) > 0, "PRETTIFY_RE RIGHT_SPACE must find matches in the input string."