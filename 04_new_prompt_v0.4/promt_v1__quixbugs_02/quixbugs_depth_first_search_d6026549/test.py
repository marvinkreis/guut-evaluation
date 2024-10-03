from node import Node
from depth_first_search import depth_first_search

def test__depth_first_search_kills_mutant():
    """
    Test depth-first search with complex graph structure containing cycles.
    The test checks if the function can reach the goal node (D) from a start node (A).
    The mutant fails to mark visited nodes, resulting in infinite recursion and thus cannot return correct results.
    """
    # Create a graph with a complex structure
    node_a = Node(value='A')
    node_b = Node(value='B')
    node_c = Node(value='C')
    node_d = Node(value='D')
    
    # Create connections
    node_a.successors = [node_b]
    node_b.successors = [node_c]
    node_c.successors = [node_a, node_d]  # Creates a cycle between A, B, C and connects to D
    node_d.successors = []  # D is a terminal node

    # Test if we can reach from A to D
    output_reach_d = depth_first_search(node_a, node_d)
    assert output_reach_d is True

    # Test if we can reach from B to D
    output_reach_d_from_b = depth_first_search(node_b, node_d)
    assert output_reach_d_from_b is True

    # Test if we can reach from C to A (should also succeed due to cycle)
    output_reach_a_from_c = depth_first_search(node_c, node_a)
    assert output_reach_a_from_c is True