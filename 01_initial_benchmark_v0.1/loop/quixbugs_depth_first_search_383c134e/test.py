from depth_first_search import depth_first_search
from node import Node

def test__depth_first_search_mutant_killer():
    # Create a test graph
    node_a = Node(value='A')        # Starting node
    node_b = Node(value='B')        # Target node
    node_c = Node(value='C')        # Extra node for cycling
    
    # Constructing the graph with a cycle
    node_a.successors = [node_b]
    node_b.successors = [node_c]
    node_c.successors = [node_b]    # Cycle between B and C

    # Perform the search where goal 'B' should be reachable
    assert depth_first_search(node_a, node_b) == True, "Should find path from A to B"

    # Test with an unreachable goal 'X'
    assert depth_first_search(node_a, Node(value='X')) == False, "Should not find path to unreachable node X"