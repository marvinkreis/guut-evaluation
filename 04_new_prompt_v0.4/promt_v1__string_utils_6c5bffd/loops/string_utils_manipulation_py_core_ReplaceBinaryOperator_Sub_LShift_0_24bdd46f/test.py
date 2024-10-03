from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test the roman_encode function for specific integer inputs to validate that the
    implementation correctly encodes integers into Roman numerals. The mutant's alteration 
    in computation will cause discrepancies for inputs greater than 5.
    Specifically testing values 6, 7, and 8, which should not produce long strings.
    """
    test_cases = {
        1: 'I',
        2: 'II',
        3: 'III',
        4: 'IV',
        5: 'V',
        6: 'VI',  # Expected: VI (Mutant should produce a very long string)
        7: 'VII', # Expected: VII (Mutant should produce a very long string)
        8: 'VIII',# Expected: VIII (Mutant should produce a very long string)
        9: 'IX',  # Expected: IX
        10: 'X'   # Expected: X
    }
    
    for input_value, expected_output in test_cases.items():
        output = roman_encode(input_value)
        print(f"input: {input_value}, output: {output}, expected: {expected_output}")
        assert output == expected_output