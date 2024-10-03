from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test the roman_encode function with the input of 40.
    The baseline should return 'XL', while the mutant is expected to raise a KeyError due to an incorrect internal mapping change.
    """
    # Checking baseline output
    output = roman_encode(40)
    assert output == 'XL', f"Expected 'XL' but got '{output}'"

    # Prepare a separate check for the mutant
    try:
        # This should succeed for the baseline
        roman_encode(40)
    except KeyError:
        # This is fine for the mutant as it is expected to raise this error
        pass
    except Exception as e:
        assert False, f"Unexpected exception raised for mutant: {str(e)}"