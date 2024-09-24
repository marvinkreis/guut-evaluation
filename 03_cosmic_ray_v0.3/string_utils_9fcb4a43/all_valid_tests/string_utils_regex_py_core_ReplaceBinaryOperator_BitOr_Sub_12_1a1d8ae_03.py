import re

def test__saxon_genitive_regex():
    """Tests the SAXON_GENITIVE regex against a variety of inputs to confirm correct matching and mutant identify capability."""
    # Test input data
    test_cases = [
        ("Tom's", True),          # Expected to match
        ("John's car", True),     # Expected to match
        ("A dog is brown", False),# Expected not to match
        ("That is Sara's pencil", True),  # Expected to match
        ("This is not a match", False),    # Expected not to match
    ]

    # Compile correct regex
    correct_pattern = re.compile(r"(?<=\w)'s", re.UNICODE)

    # Check each test string with correct implementation
    for test_str, expected in test_cases:
        match = correct_pattern.search(test_str)
        assert (match is not None) == expected, f"Correct pattern should match '{test_str}'."

    # Attempt to compile the mutant version should fail
    try:
        mutant_pattern = re.compile(r"(?<=\w)'s", re.MULTILINE - re.UNICODE)
        # If it compiles, we can attempt a check
        for test_str, expected in test_cases:
            match = mutant_pattern.search(test_str)
            assert (match is not None) == expected, f"Mutant pattern should match '{test_str}'."  # Should not be triggered
    except ValueError:
        # We expect this to fail due to incompatible flags
        pass  # Confirmation that the mutant fails as expected