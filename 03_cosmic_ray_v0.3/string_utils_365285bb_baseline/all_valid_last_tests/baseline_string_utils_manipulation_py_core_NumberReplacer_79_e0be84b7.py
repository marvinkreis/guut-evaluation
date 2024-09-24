from string_utils.manipulation import prettify

def test__strip_internal_spaces_mutant_detection():
    # Craft input string designed to test internal structure and spacing
    input_string = "    This  should   be reduced    to normal   spacing.   "

    # Expected output: spaces between words should be single
    expected_output = "This should be reduced to normal spacing."

    # Apply the prettify method
    result = prettify(input_string)

    # Check if the result matches the expected output
    assert result == expected_output, f"Expected: '{expected_output}', but got: '{result}'"

# To run the test directly
if __name__ == "__main__":
    test__strip_internal_spaces_mutant_detection()
    print("Test passed successfully.")