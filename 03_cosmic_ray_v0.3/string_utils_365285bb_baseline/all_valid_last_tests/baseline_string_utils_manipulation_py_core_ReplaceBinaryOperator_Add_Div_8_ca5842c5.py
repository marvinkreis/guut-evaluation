from string_utils.manipulation import prettify

def test_prettify():
    # Test case #1: Basic sentence with leading spaces
    input_string_1 = '    the quick brown fox.'
    expected_output_1 = 'The quick brown fox.'
    
    # Validate correct formatting
    actual_output_1 = prettify(input_string_1)
    assert actual_output_1 == expected_output_1, f'Expected: "{expected_output_1}", but got: "{actual_output_1}"'

    # Test case #2: Another leading space case
    input_string_2 = '   another example where formatting is needed.  '
    expected_output_2 = 'Another example where formatting is needed.'
    
    # Validate correct formatting
    actual_output_2 = prettify(input_string_2)
    assert actual_output_2 == expected_output_2, f'Expected: "{expected_output_2}", but got: "{actual_output_2}"'

    # Test case #3: Input with excessive internal spaces
    input_string_3 = '   multiple   internal   spaces   need   fixing.  '
    expected_output_3 = 'Multiple internal spaces need fixing.'
    
    # Validate the output
    actual_output_3 = prettify(input_string_3)
    assert actual_output_3 == expected_output_3, f'Expected: "{expected_output_3}", but got: "{actual_output_3}"'

    # Test case #4: Excessive spaces leading and trailing
    edge_case_input = '       inconsistent        spacing      example   .   '
    expected_edge_case_output = 'Inconsistent spacing example.'
    
    # Run the edge case through prettify function
    actual_edge_case_output = prettify(edge_case_input)
    assert actual_edge_case_output == expected_edge_case_output, f'Expected: "{expected_edge_case_output}", but got: "{actual_edge_case_output}"'

    # Test case #5: Case with empty input
    input_empty = '   '
    expected_empty_output = ''
    
    # Checking for edge case with empty spaces
    actual_empty_output = prettify(input_empty)
    assert actual_empty_output == expected_empty_output, f'Expected: "{expected_empty_output}", but got: "{actual_empty_output}"'