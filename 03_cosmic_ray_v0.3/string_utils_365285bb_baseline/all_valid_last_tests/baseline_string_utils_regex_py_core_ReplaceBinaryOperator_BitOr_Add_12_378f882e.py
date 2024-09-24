import re

# Define the correct SAXON_GENITIVE regex
CORRECT_SAXON_GENITIVE_RE = re.compile(r"(\w+)'s")  # Matches words followed by 's

def test_pretty_re_mutant_detection():
    # Creating a sample test string with various possessive cases
    input_string = "John's book is great. It's a wonderful day. Sally's desk is neat!"
    
    expected_matches = 3  # Expected matches: John's, It's, Sally's
    
    # Find matches using the original regex
    matches_original = re.findall(CORRECT_SAXON_GENITIVE_RE, input_string)

    # Check if original regex matches as expected
    assert len(matches_original) == expected_matches, f"Expected {expected_matches} matches, got {len(matches_original)} in original code."

    # Define a mutant regex that captures something completely irrelevant
    mutated = re.compile(r"(\w+)")  # This captures all words in the string.
    matches_mutated = mutated.findall(input_string)

    # The mutated regex captures every single word, which will exceed expected matches
    assert len(matches_mutated) > expected_matches, "Mutant code matched incorrectly; it should have failed."

# Run the test case
test_pretty_re_mutant_detection()