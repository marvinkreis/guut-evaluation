from string_utils.manipulation import prettify

def test_prettify_detects_mutant():
    # Input with excessive leading, trailing, and spaces around punctuations
    input_string = '   Hello,      world      !   '
    
    # Expected output should remove leading/trailing spaces and have no extra space before punctuation
    expected_output = 'Hello, world!'

    # Running the prettify function
    output = prettify(input_string)

    # Assert that the output matches the expected value
    assert output == expected_output, f"Expected: '{expected_output}', but got: '{output}'"

    # Mutant input that improperly handles spaces and punctuation
    mutant_input_string = 'Hello,              world!   '  # Instead, make space before '!'
    expected_mutant_output = 'Hello, world!'  # Expected output ignoring the spaces

    # Running the prettify function on the mutant case
    output_mutant = prettify(mutant_input_string)

    # Check output from mutant behavior
    assert output_mutant == expected_mutant_output, f"Expected: '{expected_mutant_output}', but got: '{output_mutant}'"

    # Checking to see if mutant outputs an incorrect formatting
    incorrect_mutant_output = 'Hello, world !'  # Here it incorrectly adds a space before the '!'
    
    # Assert the mutant's output should not match this incorrect behavior
    assert output_mutant != incorrect_mutant_output, f"Output incorrectly matches mutant behavior! Got: '{output_mutant}'"