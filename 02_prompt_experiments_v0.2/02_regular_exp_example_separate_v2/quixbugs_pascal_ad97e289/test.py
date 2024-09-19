from pascal import pascal

def test__pascal():
    """The mutant fails by raising an IndexError due to incorrect range in the loop."""
    
    # Check the correct implementation
    output = pascal(5)
    assert output == [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]], "Output does not match expected Pascal's Triangle."
    
    # Now we will check if the mutant raises an IndexError
    try:
        pascal(5)
    except IndexError:
        print("Mutant raised an IndexError as expected!")