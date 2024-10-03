from string_utils.validation import is_isbn

def test_is_isbn_mutant_killing():
    """
    Test the is_isbn function using a valid ISBN-10 value. The mutant will return False because it requires both
    ISBN-10 and ISBN-13 to be valid, while the baseline will correctly return True for valid ISBN-10 inputs.
    """
    output = is_isbn('1506715214')
    assert output == True, f"Expected True, got {output}"