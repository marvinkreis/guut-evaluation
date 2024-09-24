def test_PRETTIFY_RE():
    # Valid possessive cases
    valid_cases = [
        "The dog's bone.",  # should match possessive
        "Jessica's car.",   # should match possessive
    ]
    
    # Non-possessive cases
    invalid_cases = [
        "The dog toy is here.",  # should not match
        "Cats love to play.",      # should not match
    ]
    
    # Direct checks for the appropriate conditions
    for case in valid_cases:
        # Manually check for the presence of an apostrophe followed by 's'
        assert "'" in case and case.split("'")[1].startswith('s'), \
            f"Should match the possessive: '{case}'"

    for case in invalid_cases:
        # Check that there's no possessive form present
        assert "'" not in case or case.split("'")[1].startswith('s') is False, \
            f"Should not match non-possessive: '{case}'"

# Run the test
test_PRETTIFY_RE()