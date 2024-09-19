from pascal import pascal

def test__pascal():
    """The mutant removes one element from Pascal's Triangle rows."""
    
    correct_output = pascal(5)
    assert correct_output == [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]], \
        "The output of pascal(5) should match the expected triangle."
    
    mutant_output = pascal(2)
    assert len(mutant_output) == 2, "Mutant should return two rows."
    assert len(mutant_output[1]) == 2, "Second row of mutant output should have length 2."