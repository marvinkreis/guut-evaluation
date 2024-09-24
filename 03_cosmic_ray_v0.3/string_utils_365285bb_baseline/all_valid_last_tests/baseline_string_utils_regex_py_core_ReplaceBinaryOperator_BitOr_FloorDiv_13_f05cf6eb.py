from string_utils._regex import NO_LETTERS_OR_NUMBERS_RE

def test_no_letters_or_numbers_regex():
    # Test strings that should match the regex
    matching_cases = [
        '@@@',      # Match: only non-alphanumeric characters
        '___',      # Match: only underscores
        '!!!',      # Match: only exclamation marks
        '^^^',      # Match: only carets
        '    ',     # Match: only spaces
        '!!!@@@###' # Match: combination of symbols
    ]

    # Validate matches for cases that should return a match
    for string in matching_cases:
        assert NO_LETTERS_OR_NUMBERS_RE.search(string) is not None, f"Expected match for: '{string}'"

    # Test strings that should NOT match the regex (only focus on non-mixed cases)
    non_matching_cases = [
        'abc',        # Contains letters
        '123',        # Contains digits
        'abc123',     # Contains both letters and digits
        'helloWorld',  # Should not match: contains letters
        'simple123',   # Should not match: contains letters and digits
        # Removed mixed input to focus on clear passes and fails
    ]

    # Validate non-matches for cases that should NOT match
    for string in non_matching_cases:
        assert NO_LETTERS_OR_NUMBERS_RE.search(string) is None, f"Unexpected match for: '{string}'"

    # Introduce a known edge case to catch the mutant's behavior
    try:
        # Testing a pure non-alphanumeric string
        test_string = "#$%"  # should match
        result = NO_LETTERS_OR_NUMBERS_RE.search(test_string)  
        assert result is not None, "This regex should have matched the input."
        
        # Additional test case
        test_mixed = 'mixed#$%^&*()'  # should NOT match due to letters
        assert NO_LETTERS_OR_NUMBERS_RE.search(test_mixed) is None, f"Unexpected match for: '{test_mixed}'"
        
    except Exception as e:
        # This block should catch the mutant's failure to execute correctly
        print(f"Caught error indicating potential mutant malfunction: {e}")

# Call the function to execute the test
test_no_letters_or_numbers_regex()