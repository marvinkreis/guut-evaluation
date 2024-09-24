from string_utils.manipulation import prettify

def test__prettify():
    """Testing for proper space formatting to detect mutant behavior."""
    
    # Input with excessive and irregular spaces
    test_input = "   Hello!    This is   a   test.   How do you   do?    "
    expected_output = "Hello! This is a test. How do you do?"
    
    # Assert the expected against the prettified output
    correct_output = prettify(test_input)
    assert correct_output == expected_output, "prettify must correctly format spaces and punctuation."
    
    # Complex input to trigger potential issues with the mutant
    complex_input = "   This is a   sentence  .   With  multiple   spaces   ! "
    expected_complex_output = "This is a sentence. With multiple spaces!"
    
    # check for complex input
    correct_output_complex = prettify(complex_input)

    # Expect this to also reflect the proper handling of spaces
    assert correct_output_complex == expected_complex_output, "prettify must handle complex spacing correctly."

    # Edge case to expose mutant flaws
    edge_case_input = " !   This   one   has   irregular   spacing....    "
    expected_edge_case_output = "! This one has irregular spacing...."
    
    # This should check for the final edge case
    correct_output_edge_case = prettify(edge_case_input)
    
    # Check for the anomaly here that might reveal the mutant's failure to format
    assert correct_output_edge_case == expected_edge_case_output, "prettify must return proper formatting for edge cases."

# This test is designed to reveal the mutant's incorrect behavior when handling space formatting.