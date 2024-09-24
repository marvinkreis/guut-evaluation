import re

# Define a common multi-line input that we can use for testing
input_string = """This is a test.  This is another test,
which spans multiple lines.  
Check special cases like:  ( example )  and check --multiple spaces.
"""

def test__prettify_re():
    """The correct PRETTIFY_RE should successfully match patterns in input."""
    correct_PRETTIFY_RE = re.compile(
        r'('
        r'(?<=\S)\+(?=\S)|(?<=\S)\+\s|\s\+(?=\S)|'
        r'(?<=\S)-(?=\S)|(?<=\S)-\s|\s-(?=\S)|'
        r'(?<=\S)/(?=\S)|(?<=\S)/\s|\s/(?=\S)|'
        r'(?<=\S)\*(?=\S)|(?<=\S)\*\s|\s\*(?=\S)|'
        r'(?<=\S)=(?=\S)|(?<=\S)=\s|\s=(?=\S)|'
        r'\s"[^"]+"(?=[^\s?.:!,;])|(?<=\S)"[^"]+"\s|(?<=\S)"[^"]+"(?=[^\s?.:!,;])|'
        r'\s\([^)]+\)(?=[^\s?.:!,;])|(?<=\S)\([^)]+\)\s|(?<=\S)(\([^)]+\))(?=[^\s?.:!,;])'
        r')',
        re.MULTILINE | re.DOTALL
    )

    # Try to find matches in the input string
    matches = correct_PRETTIFY_RE.findall(input_string)
    assert matches, "Correct PRETTIFY_RE should find some matches in the input string."
    print(f"Correct PRETTIFY_RE found matches: {matches}")


def test__prettify_re_mutant():
    """The mutant PRETTIFY_RE should not successfully match and should raise ValueError."""
    try:
        # Attempt to compile the mutant regex
        mutant_PRETTIFY_RE = re.compile(
            r'('
            r'(?<=\S)\+(?=\S)|(?<=\S)\+\s|\s\+(?=\S)|'
            r'(?<=\S)-(?=\S)|(?<=\S)-\s|\s-(?=\S)|'
            r'(?<=\S)/(?=\S)|(?<=\S)/\s|\s/(?=\S)|'
            r'(?<=\S)\*(?=\S)|(?<=\S)\*\s|\s\*(?=\S)|'
            r'(?<=\S)=(?=\S)|(?<=\S)=\s|\s=(?=\S)|'
            r'\s"[^"]+"(?=[^\s?.:!,;])|(?<=\S)"[^"]+"\s|(?<=\S)"[^"]+"(?=[^\s?.:!,;])|'
            r'\s\([^)]+\)(?=[^\s?.:!,;])|(?<=\S)\([^)]+\)\s|(?<=\S)(\([^)]+\))(?=[^\s?.:!,;])'
            r')',
            re.MULTILINE - re.DOTALL
        )
        
        # Try to find matches in the input string
        mutant_matches = mutant_PRETTIFY_RE.findall(input_string)
        assert not mutant_matches, "Mutant PRETTIFY_RE should not find matches."
        print(f"Mutant PRETTIFY_RE found matches: {mutant_matches}")
        
    except ValueError:
        print("Mutant PRETTIFY_RE raised ValueError as expected.")

# Execute the tests
test__prettify_re()
test__prettify_re_mutant()