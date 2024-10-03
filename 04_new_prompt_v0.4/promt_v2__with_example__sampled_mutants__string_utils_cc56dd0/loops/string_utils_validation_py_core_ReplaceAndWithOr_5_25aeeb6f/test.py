from string_utils.validation import is_camel_case

def test_is_camel_case_mutant_killing():
    """
    Test the is_camel_case function with a valid camel case string and an invalid string.
    The mutant should return True for the invalid string due to the changed logic,
    while the baseline should return False for it.
    """
    valid_output = is_camel_case("InvalidCase123")
    assert valid_output == True, f"Expected True for valid camel case, got {valid_output}"

    invalid_output = is_camel_case("invalidcase123")
    assert invalid_output == False, f"Expected False, got {invalid_output}"