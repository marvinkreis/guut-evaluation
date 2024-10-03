from string_utils._regex import PRETTIFY_RE

def test_spaces_inside_killing_mutant():
    """
    Test the SPACES_INSIDE regex to verify it handles quoted text 
    that spans multiple lines correctly. The mutant will raise a 
    ValueError due to incompatible regex flags, while the baseline 
    functionality will return a match object.
    """
    text = '"This is a quoted text\nthat spans multiple lines."'
    match = PRETTIFY_RE['SPACES_INSIDE'].search(text)
    assert match is not None, "Expected a match, but got None"