from string_utils.manipulation import roman_encode

def test__roman_encode():
    # Testing for a value that should return 'M' (1000) to check the encoding of thousands correctly.
    output = roman_encode(1000)
    expected_output = 'M'
    assert output == expected_output, f"Expected {expected_output}, but got {output}"

    # Testing for a value that should return 'MM' (2000) to check the encoding of thousands correctly.
    output = roman_encode(2000)
    expected_output = 'MM'
    assert output == expected_output, f"Expected {expected_output}, but got {output}"

    # Testing for a value that should return 'MMM' (3000) to check the encoding of thousands correctly.
    output = roman_encode(3000)
    expected_output = 'MMM'
    assert output == expected_output, f"Expected {expected_output}, but got {output}"

    # Testing for a value that should return 'MMMCMXCIX' (3999) to check the encoding of max limit correctly.
    output = roman_encode(3999)
    expected_output = 'MMMCMXCIX'
    assert output == expected_output, f"Expected {expected_output}, but got {output}"