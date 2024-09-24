from string_utils.manipulation import prettify

def test__prettify():
    input_string = "Hello   World!"
    # Expected output after prettifying the input string
    expected_output = "Hello World!"
    
    # Running the original function
    result = prettify(input_string)
    
    # The result should match the expected output when using the original code
    assert result == expected_output, f"Expected '{expected_output}', but got '{result}'"
    
    # With the mutant code, we expect the function to fail or produce an incorrect type/output.
    # Since we cannot directly test the behavior of the mutant here, we can assume the test will
    # be run against the mutant by setting up appropriate context or state checks

# Executing test
test__prettify()