from powerset import powerset

def test__powerset():
    """ The mutant fails to behave like the original when calculating the powerset. """
    
    # Check with various inputs
    for input_data in [['a'], ['a', 'b'], ['a', 'b', 'c'], []]:
        correct_output = powerset(input_data)
        mutant_output = powerset(input_data)  # Call mutant's version here.
        
        assert correct_output == mutant_output, f"Test failed for input {input_data}: {correct_output} != {mutant_output}"
        assert [] in correct_output, f"Correct output for input {input_data} should contain the empty subset."