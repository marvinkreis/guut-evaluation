from string_utils.manipulation import roman_encode

def test__roman_encode():
    """Test cases targeting the mutant's encoding logic."""
    
    # For 8, it should generate 'VIII'
    output_8 = roman_encode(8)
    assert output_8 == "VIII", "roman_encode should return 'VIII' for the input 8"
    
    # For 4, it should generate 'IV'
    output_4 = roman_encode(4)
    assert output_4 == "IV", "roman_encode should return 'IV' for the input 4"