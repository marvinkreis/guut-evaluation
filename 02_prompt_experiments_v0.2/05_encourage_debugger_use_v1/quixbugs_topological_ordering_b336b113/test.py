from topological_ordering import topological_ordering
from node import Node

def test__topological_ordering():
    """The mutant fails when the input graph has multiple dependencies. The correct order must contain ['A', 'B', 'C', 'D']."""
    # Creating a complex graph
    node_a = Node(value='A')  # No incoming nodes
    node_b = Node(value='B', incoming_nodes=[node_a])  # Incoming from A
    node_c = Node(value='C', incoming_nodes=[node_a])  # Incoming from A
    node_d = Node(value='D', incoming_nodes=[node_b])  # Incoming from B
    
    node_a.outgoing_nodes = [node_b, node_c]
    node_b.outgoing_nodes = [node_d]
    node_c.outgoing_nodes = []

    # List of nodes
    nodes = [node_a, node_b, node_c, node_d]

    output = topological_ordering(nodes)
    assert output == [node_a, node_b, node_c, node_d], "Topological ordering must maintain the correct order for dependencies."