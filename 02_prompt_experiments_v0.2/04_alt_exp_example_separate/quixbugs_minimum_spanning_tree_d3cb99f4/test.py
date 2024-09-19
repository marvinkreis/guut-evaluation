from minimum_spanning_tree import minimum_spanning_tree

def test__minimum_spanning_tree():
    """If the mutant has an error while processing the minimum spanning tree,
    it will not match the correct output on the given input graph."""
    
    input_graph = {
        (1, 2): 10,
        (2, 3): 15,
        (3, 4): 10,
        (1, 4): 10
    }
    
    correct_output = minimum_spanning_tree(input_graph)
    expected_output = {(1, 2), (3, 4), (1, 4)}
    assert correct_output == expected_output, f"Expected {expected_output}, but got {correct_output}"