from node import Node
from topological_ordering import topological_ordering

def test__topological_ordering():
    """
    Test the topological ordering with an intricate directed graph. The correct behavior of the topological ordering 
    will ensure that nodes are returned in a sequence respecting their dependencies. The mutant will not return the 
    correct order due to its flawed check of outgoing nodes, resulting in an assertion error. 
    """
    
    # Define nodes first
    a = Node(value='A')
    b = Node(value='B')
    c = Node(value='C')
    d = Node(value='D')
    e = Node(value='E')

    # Set up their relationships
    a.outgoing_nodes = [b, c]
    b.incoming_nodes = [a]
    b.outgoing_nodes = [d]
    c.incoming_nodes = [a]
    c.outgoing_nodes = [d]
    d.incoming_nodes = [b, c]
    d.outgoing_nodes = [e]
    e.incoming_nodes = [d]
    
    nodes = [a, b, c, d, e]

    result = topological_ordering(nodes)
    expected_order = ['A', 'B', 'C', 'D', 'E']

    print(f"Ordered nodes: {[node.value for node in result]}")
    assert [node.value for node in result] == expected_order, "The ordering did not match expected output."