from string_utils.manipulation import roman_encode

def test_roman_encode_detect_mutant():
    # Test an input that directly checks the encoding of 4
    input_number = 4
    expected_output_correct = 'IV'  # Correct encoding for 4
    actual_output = roman_encode(input_number)
    assert actual_output == expected_output_correct, f"Expected {expected_output_correct}, but got {actual_output}."