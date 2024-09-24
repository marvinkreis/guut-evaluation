from string_utils._regex import NO_LETTERS_OR_NUMBERS_RE

def test_no_letters_or_numbers_regex():
    # Test inputs that should match (valid cases)
    matching_inputs = [
        "!!!",                  # Only special characters
        "@#$%^&*()",            # All special characters
        "    ",                 # Whitespace only
        "\t\t",                 # Tabs only
        "\n\n"                  # Newlines only
    ]
    
    # Test inputs that should not match (invalid cases)
    non_matching_inputs = [
        "abc",                  # Contains letters
        "123",                  # Contains numbers
        "hello world",          # Plain text
        "   a",                 # Spaces plus a letter
        "4score",               # Alphanumeric
        "text!",                # Contains letters
        "    A",                # Spaces plus a letter
        "!! _ !!",              # Special characters with an underscore
        "text 123",             # Alphanumeric
        "5",                    # Single digit
    ]

    # Check that all matching inputs pass
    for input_string in matching_inputs:
        result = NO_LETTERS_OR_NUMBERS_RE.fullmatch(input_string)
        assert result is not None, f"Matching failed for: '{input_string}'"

    # Check that all non-matching inputs fail
    for input_string in non_matching_inputs:
        result = NO_LETTERS_OR_NUMBERS_RE.fullmatch(input_string)
        assert result is None, f"Non-matching failed for: '{input_string}'"

# Ensure this is executed as a standalone script
if __name__ == "__main__":
    test_no_letters_or_numbers_regex()