from depth_first_search import depth_first_search
from node import Node

def test__depth_first_search():
    """The mutant does not track visited nodes, causing infinite recursion for unreachable nodes in a cyclic graph."""
    
    # Create a cyclic graph
    node_a = Node(value='A')
    node_b = Node(value='B')
    node_c = Node(value='C')

    # Set up successors to form a cycle
    node_a.successors = [node_b]
    node_b.successors = [node_c]
    node_c.successors = [node_a]  # This creates a cycle back to A

    # Create an unreachable node D
    node_d = Node(value='D')

    # Check that the correct implementation can identify that D is not reachable from A
    output = depth_first_search(node_a, node_d)
    assert output is False, "Depth-first search must return False for unreachable nodes in a cyclic graph."