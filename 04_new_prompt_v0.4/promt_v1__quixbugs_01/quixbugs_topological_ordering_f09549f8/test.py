from topological_ordering import topological_ordering
from node import Node

def test__topological_ordering_final():
    """
    Test the function to differentiate between the Baseline and the Mutant.
    The configuration used ensures that the mutant incorrectly processes outgoing nodes, 
    highlighting the difference in behavior between checking incoming versus outgoing nodes.
    The baseline should return all nodes in correct topological order, while the mutant 
    will miss nodes due to its altered logic.
    """
    # Create nodes with specific conditions
    a = Node(value='A')
    b = Node(value='B', incoming_nodes=[a])
    c = Node(value='C', incoming_nodes=[a])
    d = Node(value='D', incoming_nodes=[b]) 

    # Define outgoing nodes
    a.outgoing_nodes = [b, c]
    b.outgoing_nodes = [d]
    c.outgoing_nodes = []
    d.outgoing_nodes = []

    nodes = [a, b, c, d]

    output = topological_ordering(nodes)
    
    # Get the values for comparison
    output_values = [node.value for node in output]
    expected_baseline_output = ['A', 'B', 'C', 'D']
    expected_mutant_output = ['A', 'C']

    assert output_values == expected_baseline_output, f"Expected {expected_baseline_output} but got {output_values}"