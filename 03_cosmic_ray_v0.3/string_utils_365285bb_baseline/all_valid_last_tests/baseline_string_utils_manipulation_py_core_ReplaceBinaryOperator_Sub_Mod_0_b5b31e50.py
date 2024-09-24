from string_utils.manipulation import roman_encode

def test__roman_encode_mutant():
    """
    Test the roman_encode function to clearly identify discrepancies between 
    the correct encoding implementation and the mutant's behavior.
    """
    
    # Create test cases that should work correctly with the original logic
    expected_outputs = {
        6: 'VI',     # Should output 'VI'
        7: 'VII',    # Should output 'VII'
        8: 'VIII',   # Should output 'VIII'
        1: 'I',      # Expected for edge case
        5: 'V',      # Middle numeral check
        9: 'IX',     # Another valid output
        10: 'X',     # Check more valids
        3999: 'MMMCMXCIX'  # Edge high numeral check
    }
    
    for input_value, expected_output in expected_outputs.items():
        actual_output = roman_encode(input_value)
        # Assert that we get the correct value for each test case.
        assert actual_output == expected_output, f"Expected '{expected_output}' for {input_value}, but got '{actual_output}'"

    # Testing for exceptions on out of bound inputs
    try:
        roman_encode(0)
        assert False, "[FAIL] Expected ValueError for input 0"
    except ValueError:
        pass  # This is expected

    try:
        roman_encode(4000)
        assert False, "[FAIL] Expected ValueError for input 4000"
    except ValueError:
        pass  # This is expected

# Ensure to run this function to identify expected behavior.