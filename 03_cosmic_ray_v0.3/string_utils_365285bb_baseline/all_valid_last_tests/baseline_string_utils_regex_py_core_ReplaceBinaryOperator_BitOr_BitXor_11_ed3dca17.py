from string_utils._regex import PRETTIFY_RE

def test_PRETTIFY_RE():
    # This test string includes clear quoted content and parentheses
    test_string = '''"This is a quoted text"
and (this is text in brackets)
and more text "within quotes" 
and (final bracketed text).'''

    # The expected matches are straightforward
    expected_matches = [
        "This is a quoted text",
        "this is text in brackets",
        "within quotes",
        "final bracketed text"
    ]

    # Apply the regex to find matches
    matches = PRETTIFY_RE['SPACES_INSIDE'].findall(test_string)

    # Validate the number of matches found
    assert len(matches) == len(expected_matches), f"Expected {len(expected_matches)} matches but got {len(matches)}: {matches}"

    # Check if each expected match is present
    for expected in expected_matches:
        found = [m for m in matches if expected in m]  # Filter matches
        assert found, f"Expected match '{expected}' not found in {matches}"  # Check for presence

    # Now let's create a second case that should expose the mutant
    mutant_test_string = '''"Quoted text that spans
multiple lines"
and (capture this correctly)'''

    # Apply the regex to the mutant test string
    mutant_matches = PRETTIFY_RE['SPACES_INSIDE'].findall(mutant_test_string)

    # The expected behavior is that we should have 2 matches if the regex works correctly.
    # The mutant may not capture correctly due to mismanaged newline handling.
    assert len(mutant_matches) == 2, f"Expected 2 matches but got {len(mutant_matches)}: {mutant_matches}"

# Execute the test
test_PRETTIFY_RE()