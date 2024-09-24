from string_utils.manipulation import roman_encode

def test_roman_encode():
    # Test case that should produce 'XXXVIII' for 38
    correct_output = "XXXVIII"
    output = roman_encode(38)
    
    assert output == correct_output, f"Expected {correct_output}, but got {output}"
    
    # Test case that should produce 'X' for 10
    correct_output_10 = "X"
    output_10 = roman_encode(10)
    
    assert output_10 == correct_output_10, f"Expected {correct_output_10}, but got {output_10}"
    
    # Test case that should produce 'CXXX' for 130
    correct_output_130 = "CXXX"
    output_130 = roman_encode(130)
    
    assert output_130 == correct_output_130, f"Expected {correct_output_130}, but got {output_130}"