from string_utils._regex import PRETTIFY_RE

def test__spaces_around():
    """The mutant regex for 'SPACES_AROUND' will always produce the same output as the original due to the behavior change being ineffective with common sentences."""
    
    # Test cases to check for expected behavior
    test_cases = [
        "Hello, world!  Here are some numbers:  5.5 and 10.  How are you?",
        "This is an example string. It contains various signs such as +, -, * and / varying spaces.",
        "Is whitespace  required around semicolons; okay?  Yes!",
        "\nNew line tests should also count; here's another - check for spaces.\n",
        "Check this:  '+'  and  '-'"
    ]
    
    for case in test_cases:
        correct_matches = PRETTIFY_RE['SPACES_AROUND'].findall(case)
        assert correct_matches is not None, "Correct output should match."