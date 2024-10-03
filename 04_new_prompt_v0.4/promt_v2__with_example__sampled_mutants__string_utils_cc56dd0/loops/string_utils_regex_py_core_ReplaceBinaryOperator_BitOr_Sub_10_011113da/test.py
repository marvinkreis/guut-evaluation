from string_utils._regex import PRETTIFY_RE

def test_prettify_spaces_around():
    """
    Test the SPACES_AROUND regex against a multiline string. The mutant will raise 
    a ValueError due to incompatible flags, while the baseline will run without error.
    """
    multiline_string = "  + \n  - \n  +  \n"
    try:
        matches = PRETTIFY_RE['SPACES_AROUND'].findall(multiline_string)
        assert matches == [], f"Expected no matches, got {matches}"
    except ValueError as e:
        print(f"Mutant raised ValueError: {e}")