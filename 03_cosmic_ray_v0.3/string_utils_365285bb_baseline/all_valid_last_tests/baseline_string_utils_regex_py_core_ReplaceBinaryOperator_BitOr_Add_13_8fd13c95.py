import re

# Original regex pattern (correct usage)
NO_LETTERS_OR_NUMBERS_RE = re.compile(r'[^\w\d]+|_+', re.IGNORECASE | re.UNICODE)

# Mutant regex (flawed flag combination)
NO_LETTERS_OR_NUMBERS_RE_MUTANT = re.compile(r'[^\w\d]+|_+', re.IGNORECASE + re.UNICODE)

def test__no_letters_or_numbers_re():
    # Each test case: input string, expected (original, mutant)
    test_cases = [
        ("abc123", ([], [])),                                     # No matches expected
        ("abc@123", (['@'], ['@'])),                             # Special char match
        ("_underscore_", (['_', '_'], ['_', '_'])),             # Each underscore distinct
        ("____", (['____'], ['____'])),                           # Group of underscores
        ("abc__123!!!", (['__', '!!!'], ['__', '!!!'])),        # Examine underscores & specials
        ("", ([], [])),                                           # Empty string
        ("123_456", (['_'], ['_'])),                             # Matches single underscore
        ("___@___", (['___', '@', '___'], ['___', '@', '___'])), # Series with separator
        ("hello world!", ([' ', '!'], [' ', '!'])),              # Check spaces and punctuation
        ("café", ([], [])),                                    # Check accented characters
    ]

    # Running tests
    for test_input, (expected_original, expected_mutant) in test_cases:
        # Check original regex
        original_matches = NO_LETTERS_OR_NUMBERS_RE.findall(test_input)
        assert original_matches == expected_original, f"Expected {expected_original} for input '{test_input}' in original regex but got: {original_matches}"

        # Check mutant regex
        mutant_matches = NO_LETTERS_OR_NUMBERS_RE_MUTANT.findall(test_input)
        assert mutant_matches == expected_mutant, f"Expected {expected_mutant} for input '{test_input}' in mutant regex but got: {mutant_matches}"

# Run the test
test__no_letters_or_numbers_re()