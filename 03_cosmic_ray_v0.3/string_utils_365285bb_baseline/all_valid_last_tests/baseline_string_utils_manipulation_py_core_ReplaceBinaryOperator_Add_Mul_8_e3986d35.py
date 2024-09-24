from string_utils.manipulation import prettify

def test__string_formatter_numeric_special_handling():
    # Input string with a mixture of leading/trailing spaces and numbers
    input_string = '   1234  has    spaces   and     special & characters!     '

    # For the correct implementation, we expect:
    expected_output_correct = '1234 has spaces and special & characters!'

    # Run the prettify function
    actual_output = prettify(input_string)

    # Assert whether the actual output matches the expected outcome.
    assert actual_output == expected_output_correct, f"Expected '{expected_output_correct}', but got '{actual_output}'"

    # Mutant expectation if it fails to clean up spaces properly.
    expected_output_mutant = '   1234  has    spaces   and     special & characters!     '  # Mutant would retain spaces

    # Uncomment when testing against the mutant implementation
    # assert prettify(input_string) == expected_output_mutant, f"Expected '{expected_output_mutant}', but got '{actual_output}'"