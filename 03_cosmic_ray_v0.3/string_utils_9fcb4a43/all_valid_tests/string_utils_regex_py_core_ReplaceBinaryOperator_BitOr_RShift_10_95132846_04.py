from string_utils._regex import PRETTIFY_RE

def test__PRETTIFY_RE():
    """Tests the REGEX behavior under mutated conditions."""
    
    # Test case with clear duplicates from spaces
    test_string = "Normal   text with duplicates    present."
    
    correct_matches = PRETTIFY_RE['DUPLICATES'].findall(test_string)
    assert len(correct_matches) > 0, "Expected matches should be found for normal text."

    # A single test case that should yield no matches
    single_case_no_match = "No duplicates here."
    no_match = PRETTIFY_RE['DUPLICATES'].findall(single_case_no_match)
    assert len(no_match) == 0, "On a normal case with no duplicates, matches should yield zero."

    # Edge Case: Testing for empty adjacent line matches
    edge_case_string = "\n\n\n   \nSomething should match here!"
    edge_case_matches = PRETTIFY_RE['DUPLICATES'].findall(edge_case_string)
    assert len(edge_case_matches) > 0, "Edge case regex should yield matches for whitespace lines."

    # Now to enforce an invalid match expectation.
    invalid_case_string = "Just a text without spacing!"
    # Here we replace the total matches against the assumptions when spaces are minimal.
    edge_case_invalid_matches = PRETTIFY_RE['DUPLICATES'].findall(invalid_case_string)
    
    # Enforcing the assumption that the regex should yield no results:
    assert len(edge_case_invalid_matches) == 0, "The regex should find no matches for an invalid case."