from string_utils.manipulation import roman_encode

def test_roman_encode_mutant_killing():
    """
    Test the roman_encode function for values 6, 7, and 8. The baseline will return 
    valid Roman numeral strings for these inputs, while the mutant will return incorrect values
    due to the modified logic in encoding function.
    """
    outputs = {
        6: roman_encode(6), 
        7: roman_encode(7), 
        8: roman_encode(8)
    }
    expected_outputs = {
        6: 'VI', 
        7: 'VII', 
        8: 'VIII'
    }
    
    for value, expected in expected_outputs.items():
        assert outputs[value] == expected, f"Expected {expected} for {value}, got {outputs[value]}"