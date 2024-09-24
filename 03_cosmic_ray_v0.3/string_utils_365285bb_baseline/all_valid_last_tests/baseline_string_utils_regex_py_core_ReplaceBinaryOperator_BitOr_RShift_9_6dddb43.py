import re

# Define the PRETTIFY_RE dictionary with the correct regex
PRETTIFY_RE = {
    'UPPERCASE_AFTER_SIGN': re.compile(r'(?<=[.?!]\s)([A-Z])')  # Correct regex
}

# Simulating a different mutant behavior for the regex
MUTANT_PRETTIFY_RE = {
    'UPPERCASE_AFTER_SIGN': re.compile(r'(?<=[.?!]\s)([a-z])')  # This one captures lowercase instead
}

def test_PRETTIFY_RE():
    # Test case with punctuation and expected uppercase letters
    test_string = "Hello! This is a Test. What is that? Another Example."

    # Use the regex to find matches in the string
    correct_matches = PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].findall(test_string)
    mutant_matches = MUTANT_PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].findall(test_string)

    # Print the matches found for debugging purposes
    print(f"Correct Matches found: {correct_matches}")
    print(f"Mutant Matches found: {mutant_matches}")

    # Expected uppercase letters based on the given case
    expected_letters = {'T', 'A', 'W'}  # 'This', 'Another', 'What'

    # Checking correct code:
    assert len(correct_matches) >= len(expected_letters), "Not enough uppercase letters matched in correct code."

    # Ensure that all expected uppercase letters are found
    for letter in expected_letters:
        assert letter in correct_matches, f"Expected uppercase letter '{letter}' was not found."

    # Now verify the mutant behavior, expecting no uppercase matches
    assert len(mutant_matches) == 0, "The mutant regex should yield no valid uppercase matches; it is functioning incorrectly."

# Execute the test
test_PRETTIFY_RE()