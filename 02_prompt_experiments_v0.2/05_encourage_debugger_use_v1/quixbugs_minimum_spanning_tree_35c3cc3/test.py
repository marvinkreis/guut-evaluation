from minimum_spanning_tree import minimum_spanning_tree

def test__minimum_spanning_tree():
    """The mutant would cause a RuntimeError while merging groups in the MST algorithm."""
    complex_graph = {
        (1, 2): 10,
        (1, 3): 6,
        (1, 4): 5,
        (2, 3): 15,
        (3, 4): 4,
        (2, 4): 8
    }
    
    # Expecting correct output which would construct a valid MST 
    output = minimum_spanning_tree(complex_graph)
    assert output == {(2, 4), (3, 4), (1, 4)}, "Minimum spanning tree does not match expected output"