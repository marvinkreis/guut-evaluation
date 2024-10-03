from string_utils.manipulation import roman_encode

def test__roman_encode_value_error():
    """
    Test the upper bound of the roman_encode function. The Baseline should raise a ValueError for the input '4000',
    while the Mutant may either fail silently or raise an error due to incorrect handling of inputs.
    """
    try:
        roman_encode(4000)
        output = "No error raised"
    except ValueError as e:
        output = str(e)

    assert output == "Input must be >= 1 and <= 3999"