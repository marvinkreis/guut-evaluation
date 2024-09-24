from string_utils.manipulation import prettify

def test__prettify():
    # Test input with inconsistent spacing
    input_string = "  Hello,   world! How are you?   "
    
    # Expected output if the implementation is correct
    expected_output = "Hello, world! How are you?"

    # Call the prettify function on the input string
    result = prettify(input_string)

    # Assert the result matches the expected output for the correct implementation
    assert result == expected_output, f"Expected '{expected_output}', but got '{result}'"

    # Define an input string that will expose the mutant
    mutant_input = "   The   quick   brown   fox   jumps over the lazy dog.   "
    
    # The expected output from the correct implementation
    expected_output_correct = "The quick brown fox jumps over the lazy dog."

    # Process the input through prettify
    correct_result = prettify(mutant_input)

    # Assert that the correct implementation gives the expected result
    assert correct_result == expected_output_correct, "The correct implementation did not format properly."

    # Predicting what the mutant would output (due to replacing + with /)
    # A typical mutant behavior could yield an unexpected format
    mutant_incorrect_output = "The/quick brown fox jumps over the lazy dog."  # Sample unexpected output

    # Ensure that the correct result is not equal to the mutant's output
    assert correct_result != mutant_incorrect_output, "Mutant should produce different output."

# Run the test function
test__prettify()