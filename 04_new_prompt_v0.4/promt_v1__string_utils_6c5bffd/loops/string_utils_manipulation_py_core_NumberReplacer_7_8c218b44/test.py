from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test the roman_encode function to check for proper encoding of numbers 
    that should involve 'L' (50) in Roman numeral representation. Due to 
    the mutant's incorrect mapping, these inputs should trigger a KeyError 
    on the mutant while succeeding on the baseline implementation.
    """
    # Choose test values that will yield 'L' in the correct encoding
    test_values = [50, 100, 150]
    expected_baseline = ['L', 'C', 'CL']  # Expected output for Baseline
    outputs = [roman_encode(value) for value in test_values]

    for value, output, expected in zip(test_values, outputs, expected_baseline):
        assert output == expected  # This should pass on the baseline

    # Attempt the same outputs on the mutant should fail due to incorrect mapping
    for value in test_values:
        try:
            roman_encode(value)
        except KeyError:
            print(f"KeyError correctly raised for input: {value} on mutant.")