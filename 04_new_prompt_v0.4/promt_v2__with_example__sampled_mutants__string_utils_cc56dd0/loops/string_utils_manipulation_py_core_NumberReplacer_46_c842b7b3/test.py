from string_utils.manipulation import roman_encode

def test_roman_encode_mutant_killing():
    """
    Test the roman_encode function with an input of 4000.
    The baseline will raise a ValueError since 4000 is out of the acceptable range (1 to 3999).
    The mutant should attempt to process it and raise a KeyError.
    This test should pass for baseline and fail for the mutant.
    """
    # Expecting ValueError from baseline
    try:
        output = roman_encode(4000)
        assert output is not None, "Expected an exception to be raised, but got output instead."
    except ValueError as ve:
        print(f"Baseline ValueError: {ve}")  # Confirming the baseline behavior. 
        return  # Exit the test correctly here since baseline behavior is as expected.
    except KeyError as ke:
        raise AssertionError(f"Mutant raised KeyError: {ke}")  # This indicates the mutant is misbehaving.
    
    # If we reach here, it means no exceptions occurred which is not expected for the mutant.
    raise AssertionError("Expected ValueError for baseline or KeyError for mutant, but neither occurred.")