from string_utils._regex import PRETTIFY_RE

def test__prettify_re():
    # Test case to ensure the regex behaves correctly with a matching string
    test_string = 'This is a "quoted text" example.'
    match = PRETTIFY_RE['SPACES_INSIDE'].search(test_string)
    assert match is not None, "The regex should match quoted text."

    # Modify the string to ensure the mutant causes a failure
    test_string_mutant = 'This is a "quoted text" example.'
    match_mutant = PRETTIFY_RE['SPACES_INSIDE'].search(test_string_mutant)
    assert match_mutant is not None, "The mutated regex should not match."

    # Check if the mutant is causing a problem by trying a string that it should match
    incorrect_string = '"quoted" text'  # this should match the original but may fail in mutant
    match_incorrect = PRETTIFY_RE['SPACES_INSIDE'].search(incorrect_string)
    assert match_incorrect is not None, "The regex should still match the quoted text in this case."