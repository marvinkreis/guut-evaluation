from topological_ordering import topological_ordering
from node import Node

def test__topological_ordering():
    """If the condition checks outgoing_nodes instead of incoming_nodes in the topological_ordering,
    it will provide an incorrect result with missing nodes."""

    # Create nodes for a test case
    node_a = Node(value='A')
    node_b = Node(value='B')
    node_c = Node(value='C')
    node_d = Node(value='D')

    # Define relationships
    node_a.outgoing_nodes = [node_b, node_c]  # A -> B and A -> C
    node_b.incoming_nodes = [node_a]            # B <- A
    node_c.incoming_nodes = [node_a]            # C <- A
    node_d.incoming_nodes = [node_b, node_c]    # D <- B,C
    
    # Configure outgoing nodes
    node_b.outgoing_nodes = [node_d]  # B -> D
    node_c.outgoing_nodes = [node_d]  # C -> D

    # List of nodes
    nodes = [node_a, node_b, node_c, node_d]

    # Execute topological ordering
    output = topological_ordering(nodes)

    # Check that output contains all node instances; we check values for reference
    output_values = [node.value for node in output]
    expected_values = [node.value for node in nodes]

    assert set(output_values) == set(expected_values), "Output must contain all nodes."
    
    # Additionally check for total count
    assert len(output) == len(nodes), "Output must have the same number of nodes as input."
    
    # Check the correct ordering of dependencies
    assert output.index(node_a) < output.index(node_b), "'A' must come before 'B'."
    assert output.index(node_a) < output.index(node_c), "'A' must come before 'C'."
    assert output.index(node_b) < output.index(node_d), "'B' must come before 'D'."
    assert output.index(node_c) < output.index(node_d), "'C' must come before 'D'."

    print("Test passed!")