from string_utils._regex import PRETTIFY_RE

def test_saxon_genitive_mutant_killing():
    """
    Test the SAXON_GENITIVE regex pattern. The baseline will compile and work 
    with the regex, while the mutant will cause an OverflowError due to a 
    bitwise operation issue.
    """
    input_string = "John's book"
    match = PRETTIFY_RE['SAXON_GENITIVE'].search(input_string)
    assert match is None, f"Expected no match, but got {match.group() if match else 'None'}"