from breadth_first_search import breadth_first_search
from node import Node

def test_breadth_first_search():
    # Create a simple directed graph
    node_a = Node(value='A')
    node_b = Node(value='B')
    node_c = Node(value='C')

    # Set successors
    node_a.successors = [node_b]
    node_b.successors = [node_c]
    node_c.successors = []

    # Test if we can reach node_c from node_a
    result = breadth_first_search(node_a, node_c)
    
    # This should pass with the correct code
    assert result is True, "Should be able to reach node_c from node_a"
    
    # Test if we can reach node_b from node_a
    result = breadth_first_search(node_a, node_b)
    
    # This should also pass with the correct code
    assert result is True, "Should be able to reach node_b from node_a"
    
    # Test if we can reach node_a from node_c (which should not work)
    result = breadth_first_search(node_c, node_a)
    
    # This should pass with the correct code
    assert result is False, "Should not be able to reach node_a from node_c"