from topological_ordering import topological_ordering
from node import Node

def test__topological_ordering():
    """The mutant changes the condition for topological ordering,
    leading to an incorrect number of output nodes."""
    
    # Create nodes for testing
    a = Node(value='A')
    b = Node(value='B', incoming_nodes=[a])
    c = Node(value='C', incoming_nodes=[a])
    d = Node(value='D', incoming_nodes=[b, c])

    # Establish outgoing relationships
    a.outgoing_nodes = [b, c]
    b.outgoing_nodes = [d]
    c.outgoing_nodes = [d]

    # Create a list of nodes
    nodes = [a, b, c, d]

    # Call the topological ordering function
    output = topological_ordering(nodes)

    # Verify the output length to check for correctness
    assert len(output) == 4, "The topological ordering must include all nodes."