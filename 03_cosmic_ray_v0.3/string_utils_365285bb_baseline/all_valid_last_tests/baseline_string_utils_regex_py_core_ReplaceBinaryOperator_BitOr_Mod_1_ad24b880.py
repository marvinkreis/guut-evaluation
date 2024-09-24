from string_utils._regex import WORDS_COUNT_RE
import re

def test_words_count_re():
    # Input string designed to check regex robustness
    test_string = "Hello, world! This is a test.  "
    
    # Attempt to compile the regex to validate it catches errors in mutant
    try:
        # Compile the regex for validity test
        compiled_pattern = re.compile(WORDS_COUNT_RE.pattern)

        # Attempt matches
        matches = compiled_pattern.findall(test_string)

        # Log matches found to inspect what regex produced
        print(f"Matches found: {matches}")

        # Define expected matches and checks
        expected_matches = ['Hello', 'world', 'This', 'is', 'a', 'test']
        expected_count = len(expected_matches)

        # Assert that we found the expected number of matches
        assert len(matches) == expected_count, f"Expected {expected_count}, but got {len(matches)}."

        # Clean up matches, checking against known values
        stripped_matches = [word.strip(' ,.!? ') for word in matches]

        # Check that all expected words were found
        for word in expected_matches:
            assert word in stripped_matches, f"Did not find the expected word: '{word}' in matches."

        # Check that there are no unexpected words
        unique_matches = set(stripped_matches)
        assert len(unique_matches) == expected_count, f"Expected {expected_count} unique words, but found {len(unique_matches)}."

        print("Test passed successfully with expected outputs.")

    except Exception as e:
        # If the regex fails at any point, we capture that along with match results
        print(f"Caught an exception indicating potential mutant: {str(e)}")
        assert False, f"Caught an exception while processing: {str(e)}"

# Execute the test
test_words_count_re()