import re
from string_utils._regex import PRETTIFY_RE

def test__PRETTIFY_RE():
    """Ensure that PRETTIFY_RE does not raise an error on import and functions correctly."""
    test_string = "This is a test with multiple    spaces!!"

    # Assert that the PRETTIFY_RE dictionary has specific expected keys we want to check.
    assert isinstance(PRETTIFY_RE['DUPLICATES'], type(re.compile(''))), "Expected 'DUPLICATES' should be a compiled regex pattern."
    assert isinstance(PRETTIFY_RE['RIGHT_SPACE'], type(re.compile(''))), "Expected 'RIGHT_SPACE' should be a compiled regex pattern."

    # We'll use one of the regex patterns to check functionality
    duplicates_pattern = PRETTIFY_RE['DUPLICATES']
    matches = duplicates_pattern.findall(test_string)

    # We are expecting it to find instances of duplicates based on defined regex
    expected_matches_count = 1  # Adjust depending on the defined regex's behavior
    assert len(matches) >= expected_matches_count, "The duplicates pattern should find at least one match."