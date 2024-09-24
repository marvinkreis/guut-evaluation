from string_utils._regex import PRETTIFY_RE

def test__spaces_inside_re():
    test_string = '"hello world" and (some text)'
    
    # This should match the quoted text "hello world" and the text in parentheses
    matches = PRETTIFY_RE['SPACES_INSIDE'].findall(test_string)

    # Check the expected behavior with the correct code
    expected_matches = ['hello world', 'some text']
    assert matches == expected_matches, f"Expected {expected_matches} but got {matches}"

    # Additional assert to ensure no matches found when spaces inside are improperly detected
    wrong_test_string = '"hello  world" and (some text)'  # introducing extra space
    wrong_matches = PRETTIFY_RE['SPACES_INSIDE'].findall(wrong_test_string)

    # The mutant's regex would likely still notice it as valid due to the incorrect flag manipulation
    assert wrong_matches != expected_matches, "The mutant should behave differently for improperly spaced text."
