from string_utils._regex import PRETTIFY_RE

def test__prettify_spaces_inside():
    """
    Test whether the 'SPACES_INSIDE' regex correctly captures quoted text. The original regex includes re.DOTALL,
    allowing it to match quoted text across multiple lines; the mutant removes this ability. Thus, providing 
    input that has multiline quoted text will be matched by the original but not the mutant.
    """
    test_string = '"This is a test\nthat spans multiple lines"'
    output = PRETTIFY_RE['SPACES_INSIDE'].search(test_string)
    assert output is not None  # Should be matched in original