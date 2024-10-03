from minimum_spanning_tree import minimum_spanning_tree

def test__minimum_spanning_tree_kill_mutant():
    """
    Test for the minimum spanning tree implementation using a general graph. The test input is a simple graph that allows us to confirm the correctness of the minimum spanning tree by using unique edge weights. 
    As the mutant incorrectly modifies the group_by_node dictionary, it will raise an error while executing, thus failing the test.
    """
    edges = {
        (1, 2): 10,
        (1, 3): 5,
        (2, 3): 15,
        (2, 4): 10,
        (3, 4): 5
    }
    
    output = minimum_spanning_tree(edges)
    assert output == {(1, 2), (1, 3), (3, 4)}