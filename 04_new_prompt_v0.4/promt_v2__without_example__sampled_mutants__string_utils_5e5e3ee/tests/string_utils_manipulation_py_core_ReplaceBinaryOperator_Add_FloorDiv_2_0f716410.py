from string_utils.manipulation import roman_encode

def test__roman_encode_edge_cases():
    """
    Test the roman_encode function for numbers that involve the digit '9'.
    Specifically, we check the encoding for 9, 19, and 29, which should return
    'IX', 'XIX', and 'XXIX', respectively. The mutant code causes a TypeError when
    trying to combine values incorrectly, leading to a different output than expected.
    """
    output_nine = roman_encode(9)
    assert output_nine == 'IX', f"Expected 'IX' but got {output_nine}"
    
    output_nineteen = roman_encode(19)
    assert output_nineteen == 'XIX', f"Expected 'XIX' but got {output_nineteen}"
    
    output_twenty_nine = roman_encode(29)
    assert output_twenty_nine == 'XXIX', f"Expected 'XXIX' but got {output_twenty_nine}"