from string_utils._regex import PRETTIFY_RE

def test__uppercase_after_sign_mutant_killer():
    """
    This test checks the functionality of the UPPERCASE_AFTER_SIGN regex.
    It expects the regex to correctly match sequences where punctuation is directly followed by an uppercase letter.
    If the mutant is active, it will raise a ValueError due to invalid regex definition.
    """
    test_cases = [
        (". this is a sentence", True),  # should match
        ("? does this match", True),     # should match
        ("! another example", True),      # should match
        (".another example", False),      # should not match, no space
        ("?lowercase starts", False),     # should not match, no uppercase
        ("? T", True),                    # should match an uppercase after ?
        (". A quick brown fox", True),    # should match an uppercase after .
        ("! When does it end?", True),    # should match uppercase W after !
        ("#HashTag", False),               # should not match because there's no preceding punctuation
    ]
    
    for input_string, expected in test_cases:
        try:
            match = PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].search(input_string)
            actual = match is not None
            print(f"Testing: '{input_string}' | Expected: {expected}, Actual: {actual}")
            assert actual == expected
        except ValueError as ve:
            print(f"ValueError encountered: {ve}. This indicates the mutant may be present.")
            assert 'UPPERCASE_AFTER_SIGN' in str(ve)