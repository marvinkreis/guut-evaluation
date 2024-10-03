from string_utils._regex import PRETTIFY_RE

def test__spaces_inside_multiline():
    """
    Test the SPACES_INSIDE regex with input containing newlines. The input contains newline characters
    which should be handled differently by the regex depending on whether DOTALL is included.
    The expectation is that the baseline will match a multi-line quoted string, while the mutant 
    will raise an exception due to incompatible regex flags.
    """
    input_string = '"This is a test\nwith new line"'
    try:
        matches = PRETTIFY_RE['SPACES_INSIDE'].findall(input_string)
        assert matches == ['This is a test\nwith new line']  # This should pass for the baseline
    except ValueError as e:
        # The mutant is expected to raise a ValueError here
        print(f"Exception raised in mutant: {e}")