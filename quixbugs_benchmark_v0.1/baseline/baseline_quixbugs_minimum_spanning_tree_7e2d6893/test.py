from minimum_spanning_tree import minimum_spanning_tree

def test_minimum_spanning_tree():
    weight_by_edge = {
        (1, 2): 10,
        (2, 3): 15,
        (3, 4): 10,
        (1, 4): 10
    }
    
    expected_result = {(1, 2), (3, 4), (1, 4)}
    
    # Test with the correct implementation
    assert minimum_spanning_tree(weight_by_edge) == expected_result
    
    # Add an additional check with a graph that has the same weights but different edges.
    weight_by_edge_2 = {
        (1, 2): 10,
        (2, 3): 15,
        (1, 3): 20,
        (3, 4): 10,
        (1, 4): 10
    }
    
    expected_result_2 = {(1, 2), (3, 4), (1, 4)}
    
    # Now testing with a modified graph
    assert minimum_spanning_tree(weight_by_edge_2) == expected_result_2

# Execute the test to demonstrate its function
if __name__ == '__main__':
    test_minimum_spanning_tree()
    print("All tests passed!")