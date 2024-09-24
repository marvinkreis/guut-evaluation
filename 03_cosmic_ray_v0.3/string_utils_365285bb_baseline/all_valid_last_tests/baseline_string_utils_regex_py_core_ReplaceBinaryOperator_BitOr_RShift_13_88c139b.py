from string_utils._regex import NO_LETTERS_OR_NUMBERS_RE

def test_NO_LETTERS_OR_NUMBERS_RE():
    # Test the behavior of the original code
    assert NO_LETTERS_OR_NUMBERS_RE.sub('', 'Hello World!') == 'HelloWorld'
    assert NO_LETTERS_OR_NUMBERS_RE.sub('', 'Python3.9') == 'Python39'
    assert NO_LETTERS_OR_NUMBERS_RE.sub('', '!@#$$%') == ''
    
    # Test with underscores, spaces, and numbers
    assert NO_LETTERS_OR_NUMBERS_RE.sub('', 'test_value_123!') == 'testvalue123'
    
    # A crafted input with a mix of spaces and special characters
    input_string = 'This is a test_string 45678!!'
    expected_output = 'Thisisateststring45678'  # Expected output after cleaning.
    
    # Check the output of the original regex
    actual_output = NO_LETTERS_OR_NUMBERS_RE.sub('', input_string)
    assert actual_output == expected_output, f"Expected {expected_output}, got {actual_output}"

    # Introduce an input to highlight mutant's behavior
    mutant_test_string = 'Should___fail__under__mutant!!'

    # The correct expected output should remove underscores and special characters
    expected_output_correct = 'Shouldfailundermutant'  
    
    # Get the actual output with the original regex
    actual_output_correct = NO_LETTERS_OR_NUMBERS_RE.sub('', mutant_test_string)
    
    # Check that it matches expected output for the original code
    assert actual_output_correct == expected_output_correct, f"Expected {expected_output_correct}, got {actual_output_correct}"

    # Now assume the mutant fails to handle underscores correctly
    # The faulty output for the mutant may look like this:
    expected_output_mutant = 'Shouldfailunder__mutant!!'  # Hypothetical output when things go wrong

    # Ensure that the correct output does NOT match the expected faulty output
    assert actual_output_correct != expected_output_mutant, (
        "The mutant code should not produce the expected faulty output!"
    )

# Execute the test
test_NO_LETTERS_OR_NUMBERS_RE()