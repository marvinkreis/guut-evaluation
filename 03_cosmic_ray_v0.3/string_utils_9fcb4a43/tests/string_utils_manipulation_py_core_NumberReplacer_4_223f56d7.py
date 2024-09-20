from string_utils.manipulation import roman_encode

def test__roman_encode():
    """Modifying the tens mapping in the Roman number encoder results in failure for inputs requiring encoding of '1'.
    For example, '11' should return 'XI' but will cause a KeyError in the mutant."""
    output = roman_encode(11)
    assert output == 'XI', "roman_encode must correctly encode 11 to 'XI'"