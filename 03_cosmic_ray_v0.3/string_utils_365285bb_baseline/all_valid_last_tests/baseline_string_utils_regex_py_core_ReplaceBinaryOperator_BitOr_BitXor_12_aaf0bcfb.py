import re

# Defining a simplified regex for possessive detection
POSSESSIVE_RE = re.compile(r"\b\w+'s\b")

def test_POSSESSIVE_RE():
    # Valid possessive cases
    valid_cases = [
        "This is John's book.",    # Should match
        "That is Mary's coat.",     # Should match
        "Anna's party was fun."      # Should match
    ]

    # Testing valid cases
    for case in valid_cases:
        print(f"Testing valid case: '{case}'")
        match = POSSESSIVE_RE.search(case)
        assert match is not None, f"This should match the possessive form: '{case}' but was not matched."

    # Invalid cases that should NOT match
    invalid_cases = [
        "This is Johns book.",      # Should NOT match - missing apostrophe
        "This is the book of John.", # Should NOT match - descriptive
        "This is Anna and Brians project."  # Should NOT match - ambiguous possessive
    ]

    # Testing invalid cases
    for case in invalid_cases:
        print(f"Testing invalid case: '{case}'")
        match = POSSESSIVE_RE.search(case)
        assert match is None, f"This should NOT match: '{case}' but was matched."

# Execute the test
test_POSSESSIVE_RE()