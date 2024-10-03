from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    This test checks the encoding of the number 1000 into a Roman numeral.
    The input '1000' should encode to 'M' in the baseline.
    The mutant changes the internal mappings and may not produce a valid output,
    resulting in a KeyError as detected in the experiment.
    """
    output = roman_encode(1000)
    assert output == 'M'