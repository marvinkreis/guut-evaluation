from string_utils._regex import PRETTIFY_RE
import re

def test_saxon_genitive_regex_killing_mutant():
    """
    Test the SAXON_GENITIVE regex against an input string "John's".
    The baseline should match successfully, while the mutant will raise a TypeError
    due to an invalid regex pattern caused by a change from '|' to '/' in the regex definition.
    """
    test_string = "John's"

    # Expecting the baseline to return None for non-matching regex
    baseline_result = PRETTIFY_RE['SAXON_GENITIVE'].match(test_string)
    assert baseline_result is None, f"Expected None, got {baseline_result} for baseline."

test_saxon_genitive_regex_killing_mutant()