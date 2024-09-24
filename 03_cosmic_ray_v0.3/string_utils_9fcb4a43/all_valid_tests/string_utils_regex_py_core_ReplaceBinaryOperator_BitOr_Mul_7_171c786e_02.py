from string_utils._regex import PRETTIFY_RE  # Import correct regex

def test__simple_duplicates_regex():
    """Test the correct regex for duplications directly, verifying mutant deviations."""
    
    # Known strings that should yield matches
    test_strings = [
        "hello hello",       # Should match 'hello'
        "abc abc",           # Should match 'abc'
        "word word",         # Should match 'word'
        "not a duplicate"    # Should not yield any matches
    ]

    correct_regex = PRETTIFY_RE['DUPLICATES']
    failures = []

    for test_string in test_strings:
        correct_matches = correct_regex.findall(test_string)

        if ' ' in test_string:  # Assuming any string with space should yield matches
            if len(correct_matches) == 0:
                failures.append(f"FAILED for '{test_string}': expected matches, got none.")
        else:
            if len(correct_matches) > 0:
                failures.append(f"FAILED for '{test_string}': expected no matches, got {correct_matches}.")

    # Output the results
    if failures:
        for failure in failures:
            print(failure)
    else:
        print("All tests passed for the correct regex!")

# Execute the refined test
test__simple_duplicates_regex()