from string_utils._regex import WORDS_COUNT_RE

def test__words_count_re():
    # Valid test cases
    valid_cases = [
        "Hello, world!",
        "Python3 is great.",
        "Test123 should match.",
        "What's up?",
    ]

    # Invalid test cases (blank or non-word characters)
    invalid_cases = [
        "!@#$%^&*()",
        "",
        "    ",
        "\t",
    ]
    
    # Test valid cases
    for case in valid_cases:
        match = WORDS_COUNT_RE.findall(case)
        assert len(match) > 0, f"Valid case failed: {case}"
    
    # Test invalid cases
    for case in invalid_cases:
        match = WORDS_COUNT_RE.findall(case)
        assert len(match) == 0, f"Invalid case failed: {case}"

# To run the test manually, just call
test__words_count_re()