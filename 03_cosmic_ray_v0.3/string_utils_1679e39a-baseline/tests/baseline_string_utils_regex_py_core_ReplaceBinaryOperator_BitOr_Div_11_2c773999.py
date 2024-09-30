from string_utils._regex import PRETTIFY_RE

def test__PRETTIFY_RE():
    """
    Test whether the 'SPACES_INSIDE' regex in the PRETTIFY_RE dictionary
    correctly matches quoted text. The input is a string containing quoted text,
    which will change behavior if the 're.DOTALL' flag is incorrectly replaced
    with a division operator, causing the regex to not match.
    """
    test_string = '"hello world"'
    matches = PRETTIFY_RE['SPACES_INSIDE'].findall(test_string)
    assert matches == ['hello world']