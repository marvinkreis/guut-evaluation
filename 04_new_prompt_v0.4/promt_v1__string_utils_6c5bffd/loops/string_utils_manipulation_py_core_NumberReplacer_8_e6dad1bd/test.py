from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test to verify if the encoding of hundreds is affected by the mutant.
    The input value 100 should yield 'C' in the correct Roman numeral encoding, 
    while the mutant should produce a different (incorrect) output due to the modified mapping.
    The input value 250 should yield 'CCL' in both versions, demonstrating correct encoding only in the baseline.
    """
    # Test 100 which should return 'C'
    output_100 = roman_encode(100)
    assert output_100 == 'C', f"Expected 'C' but got {output_100}"
    
    # Test 250 which should return 'CCL'
    output_250 = roman_encode(250)
    assert output_250 == 'CCL', f"Expected 'CCL' but got {output_250}"