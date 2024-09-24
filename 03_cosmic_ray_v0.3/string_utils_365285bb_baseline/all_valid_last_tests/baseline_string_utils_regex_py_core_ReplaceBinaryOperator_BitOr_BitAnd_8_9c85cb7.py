from string_utils._regex import PRETTIFY_RE

def test_prettify_re():
    # Printing to see what the regex is actually matching
    test_cases = [
        'This is a text (with spaces around it).',  # should match
        'This will not match(no leading space).',   # should not match
        'This should not match (no trailing space)inside.',  # should not match
        'Check (this one) and (that one).',          # should match both
        'This is an (ambiguous) case that could fail.',  # should match
        '(No lead) space here.'                       # should not match
    ]
    
    for case in test_cases:
        match = PRETTIFY_RE['LEFT_SPACE'].search(case)
        print(f"Testing string: \"{case}\"")
        print(f"Match found: {match is not None}")
        # Depending on expected outcome, you could replace this with assertion checks
        # Example: If we know certain string outcomes, we can assert them here

# Execute the diagnostic tests
test_prettify_re()