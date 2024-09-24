from string_utils.manipulation import prettify

def test__prettify():
    """Tests ensure that the mutant introduces unwanted spaces before 's in Saxon genitives."""
    
    # Different test cases, including basic and complex scenarios,
    # to reveal the discrepancies with the mutant
    test_cases = [
        ("Sarah' s book is different from Paul's car.", "Sarah's book is different from Paul's car."),
        ("Jack' s car and Tim's bike are here.", "Jack's car and Tim's bike are here."),
        ("John' s pen is nice.", "John's pen is nice."),
        ("The cat' s toy is missing.", "The cat's toy is missing."),
        ("There are many' s where this occurs. A more complex' s string!", "There are many's where this occurs. A more complex's string!")
    ]
    
    for input_string, expected_output in test_cases:
        output = prettify(input_string)

        # Check that the output is equal to expected 
        assert output == expected_output, f"Expected: '{expected_output}', but got: '{output}'"
        
        # Check for unwanted spaces around 's in Saxon genitives
        assert " s " not in output, f"prettify introduced unexpected space in '{input_string}'"