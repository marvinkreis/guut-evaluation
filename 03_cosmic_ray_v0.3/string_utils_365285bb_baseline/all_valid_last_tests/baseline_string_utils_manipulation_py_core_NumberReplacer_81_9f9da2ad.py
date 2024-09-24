from string_utils.manipulation import prettify

def test_prettify_saxon_genitive():
    # Input string that includes a Saxon genitive with a space
    input_string1 = "John' s car is fast."
    expected_output1 = "John's car is fast."  # The expected output if properly formatted

    # Execute the prettification function
    actual_output1 = prettify(input_string1)

    # Validate the outputs
    assert actual_output1 == expected_output1, f"Expected '{expected_output1}', but got '{actual_output1}'"

    # Test with another Saxon genitive case
    input_string2 = "This is Lucy' s backpack."
    expected_output2 = "This is Lucy's backpack."  # Expected output after correct formatting

    # Execute the function again
    actual_output2 = prettify(input_string2)

    # Validate the second output
    assert actual_output2 == expected_output2, f"Expected '{expected_output2}', but got '{actual_output2}'"

# Run the test function
test_prettify_saxon_genitive()