from string_utils.manipulation import roman_decode

def test__roman_decode_vii():
    """
    This test verifies the decoding of the Roman numeral 'VII', which should return 7.
    The mutant will fail this test as it alters the logic that handles values in the decoding process,
    which affects the outcome for 'VII'.
    """
    output = roman_decode('VII')
    assert output == 7