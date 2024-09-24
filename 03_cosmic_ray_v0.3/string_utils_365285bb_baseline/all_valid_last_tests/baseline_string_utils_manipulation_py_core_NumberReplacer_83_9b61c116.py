from string_utils.manipulation import prettify

def test_prettify():
    # Original input string
    input_string = "Visit https://example.com and email me at test@example.com."
    expected_output = 'Visit https://example.com and email me at test@example.com.'

    # Ensure that for the correct code, we get the expected output
    actual_output = prettify(input_string)
    assert actual_output == expected_output, f"Expected: {expected_output}, but got: {actual_output}"

    # Input designed to trigger the mutant behavior
    input_string_with_multiple_urls = "Check out https://first-url.com and then https://second-url.com!"
    expected_output_correct = 'Check out https://first-url.com and then https://second-url.com!'

    # For the original code, the expected output remains the same
    actual_output_with_multiple_urls = prettify(input_string_with_multiple_urls)
    assert actual_output_with_multiple_urls == expected_output_correct, f"Expected: {expected_output_correct}, but got: {actual_output_with_multiple_urls}"

    # The output should differ for the mutant since it selects only the last found URL
    expected_output_mutant = 'Check out https://second-url.com and then !'
    
    # For the mutant, if it selects only the last URL, it will produce an incorrect presentation
    actual_output_mutant = prettify(input_string_with_multiple_urls)
    assert actual_output_mutant != expected_output_mutant, f"Expected output to differ with mutant: {expected_output_mutant}, but got: {actual_output_mutant}"
