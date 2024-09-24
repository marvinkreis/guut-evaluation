from string_utils.manipulation import prettify

def test__ensure_left_space_only():
    # Input with multiple leading spaces followed by a distinct character phrase
    input_string = "     Hello, World!"  # Five leading spaces

    # The expected output should ideally have leading spaces scrubbed down,
    # for the original to align to "Hello, World!"
    expected_output = "Hello, World!"  # Remove all leading spaces, should only return the text itself

    # Run the prettify function on this input
    output = prettify(input_string)

    # Assert that the output matches the expected output
    # The original function should return "Hello, World!"
    # The mutant function may retain leading spaces or misinterpret behavior
    assert output == expected_output, f"Expected '{expected_output}' but got '{output}'"

# Execute to analyze how each implementation behaves uniquely.