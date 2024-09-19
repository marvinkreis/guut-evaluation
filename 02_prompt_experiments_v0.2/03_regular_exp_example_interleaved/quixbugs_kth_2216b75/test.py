from kth import kth

def test__kth():
    """Changing 'return kth(above, k - num_lessoreq)' to 'return kth(above, k)' would cause an index error 
    for certain cases where the above list is empty, leading to unexpected behavior."""
    output_correct = kth([5, 3, 8, 7], 3)
    assert output_correct == 8, "The kth-lowest element must be 8"
    
    output_mutant = kth([5, 3, 8, 7], 3)  # This should produce an IndexError for the mutant