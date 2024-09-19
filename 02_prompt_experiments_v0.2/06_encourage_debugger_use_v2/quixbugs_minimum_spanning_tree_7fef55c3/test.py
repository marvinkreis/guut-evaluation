from minimum_spanning_tree import minimum_spanning_tree

def test__minimum_spanning_tree():
    """The mutant changes how nodes are grouped and causes a runtime error when trying to update during iteration."""
    weight_by_edge = {
        (1, 2): 10,
        (2, 3): 15,
        (3, 4): 10,
        (1, 4): 10
    }
    
    # Testing that correct output is as expected and not causing an error
    output = minimum_spanning_tree(weight_by_edge)
    assert output == {(1, 2), (3, 4), (1, 4)}, "Output does not match the expected minimum spanning tree."