from string_utils.manipulation import roman_encode

def test_roman_encode():
    # Test for the encoding of the number 4
    actual_output = roman_encode(4)
    expected_output = 'IV'  # Original expected output

    # Debug output to see the captured values
    print(f"Running test for input: 4, Actual Output: {actual_output}, Expected Output: {expected_output}")

    # Assert that the output matches expected
    assert actual_output == expected_output, f"Expected '{expected_output}' for input 4, but got '{actual_output}'."

if __name__ == "__main__":
    test_roman_encode()
    print("Test passed!")  # This message will only show if no assertion fails