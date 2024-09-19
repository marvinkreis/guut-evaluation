from depth_first_search import depth_first_search
from node import Node

def test__depth_first_search():
    """The mutant omits the tracking of visited nodes, leading to infinite loops."""
    
    # Create a simple cyclic graph
    node_a = Node(value='A')
    node_b = Node(value='B')
    node_c = Node(value='C')
    
    # Setting up successors to create a cycle A -> B -> C -> A
    node_a.successors.append(node_b)
    node_b.successors.append(node_c)
    node_c.successors.append(node_a)

    # Check if the goal can be reached
    output = depth_first_search(node_a, node_c)
    assert output, "Expected to find path from A to C"