from string_utils.validation import is_slug

def test_is_slug_mutant_killing():
    """
    Test the is_slug function with an empty string. The mutant will incorrectly return True,
    while the baseline will correctly return False. This test checks that the mutant behaves differently
    from the baseline regarding empty strings.
    """
    output = is_slug('')
    assert output == False, f"Expected False, got {output}"