from string_utils._regex import PRETTIFY_RE

def test_PRETTIFY_RE():
    # Define cases that should help differentiate the two versions
    test_cases = [
        'A "Hello".',                                        # Should match
        'This is valid: "Test!"',                            # Should match due to punctuation
        '"Example"',                                        # Should match basic quoted text
        'Invalid format"quoted',                             # Should not match
        'Multi line:\n"Should match with DOTALL."',         # Crucial test for dots and multi-line
        'Leading new line: \n"Should still match."',        # Edge case for lines
        '  "Spaces before and after" and text.',            # Should match, context of quoted string
        '"Another well-formed case!"',                       # Should match
        'Yet another case that should not match here.',      # Should not match
        '\nThis sentence has leading newlines: \n"Check!"'  # Tests for leading newlines and quotes
    ]

    print("Testing the regex with adjusted cases:")
    
    for test in test_cases:
        match = PRETTIFY_RE['LEFT_SPACE'].search(test)
        print(f"Testing string: {test!r} | Match: {match is not None}")

# Execute the test
test_PRETTIFY_RE()