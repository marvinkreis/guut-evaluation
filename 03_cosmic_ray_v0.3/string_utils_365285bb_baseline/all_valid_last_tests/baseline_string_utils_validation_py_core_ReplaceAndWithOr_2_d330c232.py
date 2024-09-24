from string_utils.validation import is_decimal

def test_is_decimal():
    # Test case: Non-number with a decimal point
    input_string = "abc.def"  # Should return False for is_decimal in the original code
    output = is_decimal(input_string)
    assert output == False, f"Expected False, got {output}"

    # Test case: A valid decimal number
    input_string = "42.0"  # Should return True for is_decimal in both original and mutant
    output = is_decimal(input_string)
    assert output == True, f"Expected True, got {output}"

    # Test case: An integer (no decimal point)
    input_string = "42"  # Should return False for is_decimal in the original code
    output = is_decimal(input_string)
    assert output == False, f"Expected False, got {output}"

    # Test case: A string with spaces and a decimal (invalid input)
    input_string = "2.5 is a number"
    output = is_decimal(input_string)
    assert output == False, f"Expected False, got {output}"

    # Test case: A negative decimal number
    input_string = "-3.14"  # Should return True for is_decimal in both original and mutant
    output = is_decimal(input_string)
    assert output == True, f"Expected True, got {output}"