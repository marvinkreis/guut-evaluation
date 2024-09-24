from string_utils._regex import PRETTIFY_RE

def test__regex_duplicates_final():
    """This test checks correct detection of duplicates and ensures mutant behavior is highlighted."""
    
    correct_regex = PRETTIFY_RE['DUPLICATES']
    inputs = [
        "test test",       # Should detect duplicate 'test'
        "foo foo",         # Should detect duplicate 'foo'
        "bar",             # Should not match anything
        "hello hello not", # Should detect duplicate 'hello'
        "this this this"   # Should detect duplicate 'this'
    ]

    failed_cases = []

    for input_string in inputs:
        correct_matches = correct_regex.findall(input_string)
        expected_has_duplicates = ' ' in input_string  # If there is a space, it can imply duplicates

        if expected_has_duplicates:
            if len(correct_matches) == 0:
                failed_cases.append(f"Expected a match for '{input_string}', got none.")
        else:
            if len(correct_matches) > 0:
                failed_cases.append(f"Expected no match for '{input_string}', got {correct_matches}.")

    if failed_cases:
        for failure in failed_cases:
            print(failure)
    else:
        print("All tests passed for the correct regex!")

# Run the final test
test__regex_duplicates_final()