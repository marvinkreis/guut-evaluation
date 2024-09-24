from string_utils.manipulation import prettify

def test__prettify():
    """Testing 'prettify' function for identifying spacing issues from mutant behavior."""
    
    # Input with irregular spaces that should be prettified
    test_input = "This  is  a  test.    There are    spaces   too."
    
    # Expected output should remove excess spaces
    expected_output = "This is a test. There are spaces too."

    # Perform the prettify
    correct_output = prettify(test_input)

    # Assert that the output matches the expected output
    assert correct_output == expected_output, "prettify must correctly format spaces."

    # Tests that might trigger faulty logic from the mutant
    test_input_faulty_logic = "A sentence   with irregular spacing!  And  multiple  spaces ."
    expected_output_faulty_logic = "A sentence with irregular spacing! And multiple spaces."

    # Run the prettify method on the faulty logic input
    correct_output_faulty_logic = prettify(test_input_faulty_logic)

    # Check if the output is as expected which should highlight mutants failing
    assert correct_output_faulty_logic == expected_output_faulty_logic, "prettify must format with multiple spaces successfully."

# This test method is designed with the potential edge cases to expose the mutant's faulty behavior related to spacing.