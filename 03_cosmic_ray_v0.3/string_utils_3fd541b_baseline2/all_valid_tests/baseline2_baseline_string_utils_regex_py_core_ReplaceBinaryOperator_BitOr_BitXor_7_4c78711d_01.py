from string_utils._regex import PRETTIFY_RE

def test__prettify_re():
    # Normal input that should pass as it contains spaces and should be prettified
    input_string = "This is a test input with multiple   spaces, yes  yes!!!"
    expected_output = "This is a test input with multiple spaces, yes yes!!!"  # Spaces should be reduced

    # Prettify function to utilize pre-defined regex
    def prettify(input_str):
        # Replace duplicate spaces
        input_str = PRETTIFY_RE['DUPLICATES'].sub(' ', input_str)
        # Further prettifications could go here...
        return input_str.strip()

    # Running the prettify function on the input
    output = prettify(input_string)
    
    # Check the expected output against the actual output
    assert output == expected_output, f"Expected: '{expected_output}', but got: '{output}'"