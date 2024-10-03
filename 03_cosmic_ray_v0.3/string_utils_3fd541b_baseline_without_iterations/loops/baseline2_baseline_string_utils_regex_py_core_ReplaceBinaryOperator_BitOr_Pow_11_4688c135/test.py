from string_utils._regex import PRETTIFY_RE

def test__PRETTIFY_RE():
    # This test case checks if the PRETTIFY_RE correctly matches text inside quotes and parentheses.
    test_string = '"hello world" and (this is a test)'
    
    # Expectation: should match "hello world" and this is a test.
    matches = PRETTIFY_RE['SPACES_INSIDE'].findall(test_string)

    # The correct code should result in 2 matches.
    assert len(matches) == 2, f"Expected 2 matches, got {len(matches)}. Matches found: {matches}"
    
    # Now let's check the behavior with the mutant. The mutant changes the bitwise OR `|` to bitwise AND `**`
    # So we need to create a scenario where the PRETTIFY_RE regex pattern fails, as it won't match correctly.

test__PRETTIFY_RE()