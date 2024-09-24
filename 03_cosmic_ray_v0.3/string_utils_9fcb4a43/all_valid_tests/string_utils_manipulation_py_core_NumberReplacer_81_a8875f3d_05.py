from string_utils.manipulation import prettify

def test__prettify():
    """Tests ensure that the mutant introduces unwanted spaces before 's in Saxon genitives."""
    
    # Inputs expected to test output manipulation directly regarding Saxon genitives
    test_cases = [
        ("Sarah' s book", "Sarah's book"),
        ("Jack' s car", "Jack's car"),
        ("John' s pen", "John's pen"),
        ("The cat' s toy", "The cat's toy"),
        ("Emily' s book is here.", "Emily's book is here."),
        ("Paul' s car and Tim' s bike.", "Paul's car and Tim's bike."),
        ("There are Sarah' s dog and Tim' s cat.", "There are Sarah's dog and Tim's cat."),
    ]
    
    for input_string, expected_output in test_cases:
        output = prettify(input_string)

        # Check that the output is equal to expected 
        assert output == expected_output, f"Expected: '{expected_output}', but got: '{output}'"
        
        # Check for unwanted spaces around 's in Saxon genitives.
        # The following asserts there should NOT be a space before 's in the output.
        assert " s " not in output, f"prettify introduced unexpected space in '{input_string}'"