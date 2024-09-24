from string_utils.manipulation import prettify

def test__string_formatter_space_handling():
    # Input with multiple spaces
    input_string = 'Hello    World     this      is     a test.     '
    
    # Expected output: All extra spaces should be reduced to single spaces,
    # and the ending spaces should be trimmed
    expected_output_correct = 'Hello World this is a test.'
    
    # Run the original prettify function
    result = prettify(input_string)

    # Check the output against the expected correct output
    assert result == expected_output_correct, f"Expected '{expected_output_correct}' but got '{result}'"

    # For the mutant's output, let's assume similar logic applies but mishandled the space reduction
    expected_output_mutant = 'Hello World this is a test.'  # Expected behavior assuming it processes correctly

    # Simulate the mutant's result (in reality would be different as the implementation fails).
    mutant_result = prettify(input_string)

    # Ensure mutant and original behaviors differ substantially due to expected logic flaws
    assert mutant_result == expected_output_mutant, f"Mutant should yield '{expected_output_mutant}' but got '{mutant_result}'"

    # Test with normal input (no excessive spaces)
    normal_input = 'Just a normal line.'
    normal_output = prettify(normal_input)

    # Assert normal behavior
    assert normal_output == normal_input, f"Expected output to match input; got '{normal_output}'"

# Execute the test function.