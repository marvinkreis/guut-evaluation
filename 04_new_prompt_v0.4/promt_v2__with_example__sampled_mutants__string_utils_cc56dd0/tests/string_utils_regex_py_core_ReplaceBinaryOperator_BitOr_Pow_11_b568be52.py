from string_utils._regex import PRETTIFY_RE

def test_pretiffy_spaces_inside_mutant_killing():
    """
    Test the SPACES_INSIDE regex handling with multiline quoted text.
    The baseline should match quoted text, while the mutant results in an OverflowError.
    """
    input_string = '''"Hello, World!"
                      "This is a test"
                      "Goodbye!"'''
    matches = PRETTIFY_RE['SPACES_INSIDE'].findall(input_string)
    assert len(matches) > 0, "Expected matches for quoted text."