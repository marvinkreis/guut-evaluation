from topological_ordering import topological_ordering
from node import Node

def test_topological_ordering():
    # Create a directed acyclic graph (DAG)
    nodeA = Node(value='A')
    nodeB = Node(value='B', incoming_nodes=[nodeA])
    nodeC = Node(value='C', incoming_nodes=[nodeA])
    nodeD = Node(value='D', incoming_nodes=[nodeB])
    
    nodeA.outgoing_nodes = [nodeB, nodeC]
    nodeB.outgoing_nodes = [nodeD]
    
    nodes = [nodeA, nodeB, nodeC, nodeD]

    # We expect the order of nodes to be: A, B, C, D
    result = topological_ordering(nodes)

    # Check the order based on the values of nodes
    result_values = [node.value for node in result]

    # Assert that the order is as expected
    assert result_values == ['A', 'B', 'C', 'D']

# Run test
test_topological_ordering()