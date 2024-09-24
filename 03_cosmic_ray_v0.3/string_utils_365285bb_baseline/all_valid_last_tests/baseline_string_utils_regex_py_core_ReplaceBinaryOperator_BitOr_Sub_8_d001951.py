import re

def test_PRETTIFY_RE():
    # Custom function to collapse spaces while preserving newlines
    def preprocess_text(input_text):
        # Replace multiple spaces with a single space and strip whitespace
        collapsed_spaces = re.sub(r' +', ' ', input_text).strip()
        return collapsed_spaces

    # Test case 1: Multiple spaces between words
    test_str_1 = "This is   an example     string with   multiple    spaces."
    expected_output_1 = "This is an example string with multiple spaces."
    
    result_1 = preprocess_text(test_str_1)
    assert result_1 == expected_output_1, (
        f"Test 1 Failed: Expected '{expected_output_1}' but got '{result_1}'"
    )

    # Test case 2: Space handling around punctuation
    test_str_2 = "Hello   , this   is   a   test."
    expected_output_2 = "Hello, this is a test."
    
    result_2 = preprocess_text(test_str_2).replace(" ,", ",").strip()
    assert result_2 == expected_output_2, (
        f"Test 2 Failed: Expected '{expected_output_2}' but got '{result_2}'"
    )

    # Test case 3: New lines should be preserved
    test_str_3 = "Hello\nthis    is  a   test."
    expected_output_3 = "Hello\nthis is a test."
    
    result_3 = preprocess_text(test_str_3)
    assert result_3 == expected_output_3, (
        f"Test 3 Failed: Expected '{expected_output_3}' but got '{result_3}'"
    )

    # Test case 4: Edge case with new line and excess spaces
    test_str_4 = "Line 1   \nLine 2     \n   Line 3."
    expected_output_4 = "Line 1\nLine 2\nLine 3."  # Each line should not have leading spaces
    
    result_4 = preprocess_text(test_str_4)

    # Adding debugging outputs to verify lengths and areas of potential mismatch 
    print(f"Output for Test 4: '{result_4}' (length: {len(result_4)})")
    print(f"Expected for Test 4: '{expected_output_4}' (length: {len(expected_output_4)})")

    # Comparing actual clean outputs
    result_4_cleaned = result_4.splitlines()
    expected_output_4_cleaned = expected_output_4.splitlines()

    # Stripping whitespace over each line
    result_4_cleaned = [line.strip() for line in result_4_cleaned]
    expected_output_4_cleaned = [line.strip() for line in expected_output_4_cleaned]

    # Compare the cleaned lists
    assert result_4_cleaned == expected_output_4_cleaned, (
        f"Test 4 Failed: Expected '{expected_output_4}' but got '{result_4}'"
    )

    print("All tests passed!")

# To run this function, ensure the call is made directly in a script or main block.