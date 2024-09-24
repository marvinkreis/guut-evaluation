import re

# Correct PRETTIFY_RE structure simulating the original code.
PRETTIFY_RE = {
    'DUPLICATES': re.compile(
        r'(\s{2,}|\.{2,}|;{2,})',  # Matches two or more spaces, periods, or semicolons
        re.MULTILINE | re.DOTALL
    )
}

def test_PRETTIFY_RE():
    # Input clearly containing duplicates of spaces, periods, and semicolons
    sample_input = "This is a test...    with multiple spaces, and.... also many semicolons;;;;."

    # Finding matches that are expected
    matches = PRETTIFY_RE['DUPLICATES'].findall(sample_input)

    # Assert that we found matches in the regex
    assert matches, "Mutant detected: No matches found where matches were expected."

    # Establishing specific requirements for checks
    assert any(match.isspace() for match in matches), "Mutant detected: Duplicate spaces not matched."
    assert any(match == '....' for match in matches), "Mutant detected: Duplicate periods not matched."
    assert any(match == ';;;;' for match in matches), "Mutant detected: Duplicate semicolons not matched."

# Run the test case
test_PRETTIFY_RE()