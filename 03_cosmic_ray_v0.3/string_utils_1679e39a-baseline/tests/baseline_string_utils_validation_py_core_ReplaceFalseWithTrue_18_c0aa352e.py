from string_utils.validation import is_slug

def test__is_slug():
    """
    Test the is_slug function with an empty input string. The input is an empty string, which should return False according to the baseline. The mutant incorrectly returns True when the input is not a valid slug.
    This test case will detect the mutant by ensuring it expects False but receives True from the mutant.
    """
    output = is_slug('')
    assert output == False