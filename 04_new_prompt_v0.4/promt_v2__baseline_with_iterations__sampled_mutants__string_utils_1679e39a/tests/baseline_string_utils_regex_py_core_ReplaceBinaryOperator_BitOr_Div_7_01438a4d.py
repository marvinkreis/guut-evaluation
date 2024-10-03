from string_utils._regex import PRETTIFY_RE

def test__PRETTIFY_RE():
    """
    Test that the PRETTIFY_RE pattern correctly matches spaces around punctuation. 
    The input string contains multiple spaces before a colon, which should be matched.
    The mutant incorrectly changes '|' to '/', causing incorrect behavior and failing the test.
    """
    input_string = "Hello     : World"
    match = PRETTIFY_RE['RIGHT_SPACE'].findall(input_string)
    assert match != []