import re

def test_saxon_genitive():
    # Define the regex directly
    saxon_genitive_regex = r"(?<=\w)\'s\s|(?<=\w)\s\'s(?=\w)|(?<=\w)\s\'s\s(?=\w)"
    # Compile the regex
    pattern = re.compile(saxon_genitive_regex)

    # Test cases
    test_cases = [
        "This is Neil's pen.",         # Should match: TRUE
        "There is Alice's book.",       # Should match: TRUE
        "This pen belongs to Neil.",    # Should NOT match: FALSE
        "Is this Jane's house?",        # Should match: TRUE
        "My friend's car is red.",      # Should match: TRUE
        "Its a sunny day.",             # Should NOT match: FALSE
    ]

    # Expected outcomes
    expected_results = [
        True,  # Matches
        True,  # Matches
        False, # Doesn't match
        True,  # Matches
        True,  # Matches
        False  # Doesn't match
    ]

    # Run the tests
    for test_case, expected in zip(test_cases, expected_results):
        result = pattern.search(test_case) is not None
        assert result == expected, f"Failed for: '{test_case}'. Expected {expected} but got {result}"

    print("All regex tests passed successfully.")

# Run the regex testing function
test_saxon_genitive()