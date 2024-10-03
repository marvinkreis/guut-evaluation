import re
from string_utils._regex import PRETTIFY_RE

def test_prettify_regex_mutant_killing():
    """
    Test the SAXON_GENITIVE regex with a string that is not a possessive case. The baseline should return None,
    while the mutant will raise a ValueError due to incompatible regex flags for ASCII and UNICODE.
    """
    test_string = "Avery"
    match = PRETTIFY_RE['SAXON_GENITIVE'].search(test_string)
    assert match is None, "Expected no match, but got a match object instead"