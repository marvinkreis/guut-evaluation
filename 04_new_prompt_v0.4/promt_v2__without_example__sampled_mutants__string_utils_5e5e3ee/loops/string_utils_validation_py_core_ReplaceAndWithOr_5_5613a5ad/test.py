from string_utils.validation import is_camel_case

def test__is_camel_case_mutant_killing():
    """
    Test to ensure that 'hello world' is not recognized as camel case.
    The baseline should return false, while the mutant will return true,
    thus killing the mutant and showing the difference in behavior.
    """
    # This should be False in the baseline
    assert is_camel_case("hello world") == False